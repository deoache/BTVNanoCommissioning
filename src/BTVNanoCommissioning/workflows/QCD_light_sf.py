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

        ## Set output histogram
        syst_axis = hist.axis.StrCategory([], name="syst", growth=True)
        flav_axis = hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour")
        jpt_axis = hist.axis.Regular(300, 0, 3000, name="jpt", label="Jet $p_{T}$ [GeV]")

        self._hist_dict = {}
        self._working_points = ["L", "M", "T"]
        
        self._hist_dict["jet_pt"] = hist.Hist(syst_axis, jpt_axis, flav_axis, hist.storage.Weight())
        for wp in self._working_points:
            self._hist_dict[f"UParTAK4B_{wp}_postag_jet_pt"] = hist.Hist(syst_axis, jpt_axis, flav_axis, hist.storage.Weight())
            self._hist_dict[f"UParTAK4B_{wp}_negtag_jet_pt"] = hist.Hist(syst_axis, jpt_axis, flav_axis, hist.storage.Weight())
        

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events)
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), name)
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
            if self._year == "2024":
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
        pruned_ev["SelJet"] = event_jet[event_level]

        if not isRealData:
            flav = ak.values_astype(
                ak.where(
                    (event_jet.hadronFlavour == 0) & (event_jet.partonFlavour == 0),
                    event_jet.hadronFlavour + 1,
                    event_jet.hadronFlavour,
                ),
                int
            )
            
        for wp in self._working_points:
            wp_value = btag_wp_dict[f"{self._year}_{self._campaign}"]["UParTAK4"]["b"][wp]
            postag_mask = event_jet.btagUParTAK4B > wp_value
            negtag_mask = event_jet.btagNegUParTAK4B > wp_value
            postag_jet = event_jet[postag_mask][event_level]
            negtag_jet = event_jet[negtag_mask][event_level]
            pruned_ev[f"UParTAK4B_{wp}_postag_jet"] = postag_jet
            pruned_ev[f"UParTAK4B_{wp}_negtag_jet"] = negtag_jet

            if isRealData:
                pruned_ev[f"UParTAK4B_{wp}_postag_jet", "flav"] = ak.zeros_like(postag_jet.pt, dtype=int)
                pruned_ev[f"UParTAK4B_{wp}_negtag_jet", "flav"] = ak.zeros_like(negtag_jet.pt, dtype=int)
            else:
                pruned_ev[f"UParTAK4B_{wp}_postag_jet", "flav"] = flav[postag_mask][event_level]
                pruned_ev[f"UParTAK4B_{wp}_negtag_jet", "flav"] = flav[negtag_mask][event_level]

        if isRealData:
            pruned_ev["SelJet", "flav"] = ak.zeros_like(event_jet[event_level].pt, dtype=int)
        else:
            pruned_ev["SelJet", "flav"] = flav[event_level]
            
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

                print(f"{pruned_ev.SelJet.flav=}")
                print(f"{flatten(pruned_ev.SelJet.flav)=}")
                output["jet_pt"].fill(
                    syst,
                    flatten(pruned_ev.SelJet.pt),
                    flatten(pruned_ev.SelJet.flav),
                    weight=flatten(ak.broadcast_arrays(weight, pruned_ev.SelJet.pt)[0]),
                )
                for wp in self._working_points:
                    output[f"UParTAK4B_{wp}_postag_jet_pt"].fill(
                        syst,
                        flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt),
                        flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].flav),
                        weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt)[0]),
                    )
                    output[f"UParTAK4B_{wp}_negtag_jet_pt"].fill(
                        syst,
                        flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt),
                        flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].flav),
                        weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt)[0]),
                    )
        elif (not isRealData) and (shift_name is not None):
            weight = weights.weight()
            output["jet_pt"].fill(
                shift_name,
                flatten(pruned_ev.SelJet.pt),
                flatten(pruned_ev.SelJet.flav),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.SelJet.pt)[0]),
            )
            for wp in self._working_points:
                output[f"UParTAK4B_{wp}_postag_jet_pt"].fill(
                    shift_name,
                    flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt),
                    flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].flav),
                    weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt)[0]),
                )
                output[f"UParTAK4B_{wp}_negtag_jet_pt"].fill(
                    shift_name,
                    flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt),
                    flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].flav),
                    weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt)[0]),
                )
        elif isRealData and (shift_name is None):
            weight = weights.weight()
            output["jet_pt"].fill(
                "nominal",
                flatten(pruned_ev.SelJet.pt),
                flatten(pruned_ev.SelJet.flav),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.SelJet.pt)[0]),
            )
            for wp in self._working_points:
                output[f"UParTAK4B_{wp}_postag_jet_pt"].fill(
                    "nominal",
                    flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt),
                    flatten(pruned_ev[f"UParTAK4B_{wp}_postag_jet"].flav),
                    weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_postag_jet"].pt)[0]),
                )
                output[f"UParTAK4B_{wp}_negtag_jet_pt"].fill(
                    "nominal",
                    flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt),
                    flatten(pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].flav),
                    weight=flatten(ak.broadcast_arrays(weight, pruned_ev[f"UParTAK4B_{wp}_negtag_jet"].pt)[0]),
                )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator