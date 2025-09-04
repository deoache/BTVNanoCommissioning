import os
import copy, hist, collections, numpy as np, awkward as ak
from coffea import processor
from coffea.analysis_tools import Weights
from BTVNanoCommissioning.helpers.func import flatten, update, dump_lumi
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)
from BTVNanoCommissioning.utils.selection import jet_id, HLT_helper, btag_wp_dict

import correctionlib
import numpy as np


def normalize(array: ak.Array):
    if array.ndim == 2:
        return ak.fill_none(ak.flatten(array), np.nan)
    else:
        return ak.fill_none(array, np.nan)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2024",
        campaign="Summer24",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        addsel=False,
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

        ## Define histogram axes
        self._working_points = ["L", "M", "T"]
        syst_axis = hist.axis.StrCategory([], name="syst", growth=True)
        flav_axis = hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour")
        jpt_axis = hist.axis.Regular(300, 0, 3000, name="jpt", label="Jet $p_{T}$ [GeV]")
        b_disc_axis = hist.axis.Regular(50, 0, 1, name=f"UParTAK4BDisc", label=f"UParTAK4BDisc")
        b_discn_axis = hist.axis.Regular(50, 0, 1, name=f"UParTAK4BDiscN", label=f"UParTAK4BDiscN")
        b_pass_disc_axis = hist.axis.IntCategory([0, 1], name=f"PassUParTAK4BDisc")
        b_pass_discn_axis = hist.axis.IntCategory([0, 1], name=f"PassUParTAK4BDiscN")

        ## Set output histogram
        self._hist_dict = {}
        self._hist_dict["UParTAK4BDisc"] = hist.Hist(syst_axis, flav_axis, b_disc_axis, b_discn_axis, hist.storage.Weight())
        for wp in self._working_points:
            self._hist_dict[f"PassUParTAK4BDisc_{wp}"] = hist.Hist(syst_axis, jpt_axis, flav_axis, b_pass_disc_axis, b_pass_discn_axis, hist.storage.Weight())
        

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events)
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        output = copy.deepcopy(self._hist_dict)

        if isRealData:
            output["sumw"] = len(events)
        else:
            output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################
        # NPV
        npvsGood = ak.values_astype(events.PV.npvsGood, np.int32)
        req_npvsgood = npvsGood > 0

        ## Jet cuts
        event_jet = events.Jet[jet_id(events, self._campaign, max_eta=2.5, min_pt=20)]
        req_jets = ak.count(event_jet.pt, axis=1) >= 1

        ## HLT
        triggers = {
            "PFJet40": [45, 80],
            "PFJet60": [80, 110],
            "PFJet80": [110, 180],
            "PFJet140": [180, 250],
            "PFJet200": [250, 310],
            "PFJet260": [310, 380],
            "PFJet320": [380, 460],
            "PFJet400": [460, 520],
            "PFJet450": [520, 580],
            "PFJet500": [580, 1e7],
        }
        req_trig = np.zeros(len(events), dtype="bool")
        trigbools = {}
        for trigger, ptrange in triggers.items():
            ptmin = ptrange[0]
            ptmax = ptrange[1]
            # Require *leading jet* to be in the pT range of the trigger
            thistrigreq = (
                (HLT_helper(events, [trigger]))
                & (ak.fill_none(ak.firsts(event_jet.pt) >= ptmin, False))
                & (ak.fill_none(ak.firsts(event_jet.pt) < ptmax, False))
            )
            trigbools[trigger] = thistrigreq
            req_trig = (req_trig) | (thistrigreq)

        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
            
            # veto 2024 problematic runs
            problematic_runs = [380126, 380127, 380128]
            req_runs = np.ones(len(events), dtype="bool")
            for problematic_run in problematic_runs:
                req_runs = req_runs & (events.run != problematic_run)
            req_lumi = req_lumi & req_runs
            
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        event_level = ak.fill_none(req_lumi & req_trig & req_jets & req_npvsgood, False)
        if len(events[event_level]) == 0:
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level][:, 0]
        pruned_ev["jpt"] = pruned_ev.SelJet.pt
        pruned_ev["UParTAK4BDisc"] = pruned_ev.SelJet.btagPNetB
        pruned_ev["UParTAK4BDiscN"] = pruned_ev.SelJet.btagNegPNetB

        for wp in self._working_points:
            wp_value = btag_wp_dict[f"{self._year}_{self._campaign}"]["UParTAK4"]["b"][wp]
            pruned_ev[f"PassUParTAK4BDisc_{wp}"] = (pruned_ev.SelJet.btagUParTAK4B > wp_value)
            pruned_ev[f"PassUParTAK4BDiscN_{wp}"] = (pruned_ev.SelJet.btagNegUParTAK4B > wp_value)

        if isRealData:
            pruned_ev["flav"] = ak.zeros_like(pruned_ev.SelJet.pt, dtype=int)
        else:
            pruned_ev["flav"] = ak.values_astype(
                pruned_ev.SelJet.hadronFlavour
                + 1
                * (
                    (pruned_ev.SelJet.partonFlavour == 0)
                    & (pruned_ev.SelJet.hadronFlavour == 0)
                ),
                int,
            )

        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        if isRealData:
            if self._year == "2022":
                run_num = "355374_362760"
            elif self._year == "2023":
                run_num = "366727_370790"
            elif self._year == "2024":
                run_num = "378985_386951"

            psweight = np.zeros(len(pruned_ev))
            for trigger, trigbool in trigbools.items():
                psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trigger}_run{run_num}.json"
                if not os.path.isfile(psfile):
                    raise NotImplementedError(
                        f"Prescale weights not available for {trigger} in {self._year}. Please run `scripts/dump_prescale.py`."
                    )
                pseval = correctionlib.CorrectionSet.from_file(psfile)
                thispsweight = pseval["prescaleWeight"].evaluate(
                    pruned_ev.run,
                    f"HLT_{trigger}",
                    ak.values_astype(pruned_ev.luminosityBlock, np.float32),
                )
                psweight = ak.where(trigbool[event_level], thispsweight, psweight)
            weights.add("psweight", psweight)

        ####################
        #     Output       #
        ####################
        if (not isRealData) and (shift_name is None):
            systematics = ["nominal"] + list(weights.variations)
            for syst in systematics:
                if syst == "nominal":
                    weight = weights.weight()
                else:
                    weight = weights.weight(modifier=syst)
                    
                output["UParTAK4BDisc"].fill(
                    syst=syst,
                    flav=normalize(pruned_ev.flav),
                    UParTAK4BDisc=normalize(pruned_ev.UParTAK4BDisc),
                    UParTAK4BDiscN=normalize(pruned_ev.UParTAK4BDiscN),
                    weight=weight
                )
                for wp in self._working_points:
                    output[f"PassUParTAK4BDisc_{wp}"].fill(
                        syst=syst,
                        flav=normalize(pruned_ev.flav),
                        jpt=normalize(pruned_ev.jpt),
                        PassUParTAK4BDisc=normalize(pruned_ev[f"PassUParTAK4BDisc_{wp}"]),
                        PassUParTAK4BDiscN=normalize(pruned_ev[f"PassUParTAK4BDiscN_{wp}"]),
                        weight=weight
                    )
        elif (not isRealData) and (shift_name is not None):
            weight = weights.weight()
            output["UParTAK4BDisc"].fill(
                syst=shift_name,
                flav=normalize(pruned_ev.flav),
                UParTAK4BDisc=normalize(pruned_ev.UParTAK4BDisc),
                UParTAK4BDiscN=normalize(pruned_ev.UParTAK4BDiscN),
                weight=weight
            )
            for wp in self._working_points:
                output[f"PassUParTAK4BDisc_{wp}"].fill(
                    syst=shift_name,
                    flav=normalize(pruned_ev.flav),
                    jpt=normalize(pruned_ev.jpt),
                    PassUParTAK4BDisc=normalize(pruned_ev[f"PassUParTAK4BDisc_{wp}"]),
                    PassUParTAK4BDiscN=normalize(pruned_ev[f"PassUParTAK4BDiscN_{wp}"]),
                    weight=weight
                )
        elif isRealData and (shift_name is None):
            weight = weights.weight()
            output["UParTAK4BDisc"].fill(
                syst="nominal",
                flav=normalize(pruned_ev.flav),
                UParTAK4BDisc=normalize(pruned_ev.UParTAK4BDisc),
                UParTAK4BDiscN=normalize(pruned_ev.UParTAK4BDiscN),
                weight=weight
            )
            for wp in self._working_points:
                output[f"PassUParTAK4BDisc_{wp}"].fill(
                    syst="nominal",
                    flav=normalize(pruned_ev.flav),
                    jpt=normalize(pruned_ev.jpt),
                    PassUParTAK4BDisc=normalize(pruned_ev[f"PassUParTAK4BDisc_{wp}"]),
                    PassUParTAK4BDiscN=normalize(pruned_ev[f"PassUParTAK4BDiscN_{wp}"]),
                    weight=weight 
                )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator