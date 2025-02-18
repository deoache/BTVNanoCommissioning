import collections, numpy as np, awkward as ak
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
from BTVNanoCommissioning.utils.selection import jet_cut, HLT_helper

import correctionlib


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
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
        output = {} if self.noHist else histogrammer(events, "QCD_light_sf")

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

        ## HLT
        triggers = [
            "PFJet140",
        ]
        req_trig = HLT_helper(events, triggers)
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)
        ## Jet cuts
        event_jet = events.Jet[(abs(events.Jet.eta) < 2.4) & (events.Jet.pt > 50) & (events.Jet.jetId >= 5)]
        req_jets = ak.count(event_jet.pt, axis=1) >= 1

        event_level = ak.fill_none(req_lumi & req_trig & req_jets & req_npvsgood, False)
        if len(events[event_level]) == 0:
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        pruned_ev = events[event_level]
        pruned_ev["npvs"] = ak.values_astype(pruned_ev.PV.npvsGood, np.int32)
        pruned_ev["SelJet"] = event_jet[event_level]
        pruned_ev["jpt"] = event_jet[event_level].pt
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)
        
        pruned_ev["PNetBDisc"] = event_jet[event_level].btagPNetB
        pruned_ev["PNetCDisc"] = event_jet[event_level].btagPNetProbC
        pruned_ev["PNetBDiscN"] = event_jet[event_level].btagNegPNetB
        pruned_ev["PNetCDiscN"] = event_jet[event_level].btagNegPNetProbC
        
        pruned_ev["ParTBDisc"] = event_jet[event_level].btagRobustParTAK4B
        pruned_ev["ParTCDisc"] = event_jet[event_level].btagRobustParTAK4C
        pruned_ev["ParTBDiscN"] = event_jet[event_level].btagNegRobustParTAK4B
        pruned_ev["ParTCDiscN"] = event_jet[event_level].btagNegRobustParTAK4C
        
        pruned_ev["DeepFlavourBDisc"] = event_jet[event_level].btagDeepFlavB
        pruned_ev["DeepFlavourCDisc"] = event_jet[event_level].btagDeepFlavC
        pruned_ev["DeepFlavourBDiscN"] = event_jet[event_level].btagNegDeepFlavB
        pruned_ev["DeepFlavourCDiscN"] = event_jet[event_level].btagNegDeepFlavC
        
        if isRealData:
            pruned_ev["flav"] = ak.zeros_like(event_jet[event_level].pt, dtype=int)
        else:
            pruned_ev["flav"] = event_jet[event_level].hadronFlavour

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
            pseval = correctionlib.CorrectionSet.from_file(
                f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{triggers[0]}_run{run_num}.json"
            )
            # if 369869 in pruned_ev.run: continue
            psweight = pseval["prescaleWeight"].evaluate(
                pruned_ev.run,
                f"HLT_{triggers[0]}",
                ak.values_astype(pruned_ev.luminosityBlock, np.float32),
            )
            weights.add("psweight", psweight)

        ####################
        #     Output       #
        ####################
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
