import collections, correctionlib, numpy as np, awkward as ak, os
from coffea import processor
from coffea.analysis_tools import Weights
from BTVNanoCommissioning.helpers.func import flatten, update, dump_lumi
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writter,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)
from BTVNanoCommissioning.utils.selection import HLT_helper, jet_id, btag_wp_dict


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

        output = {}
        if not self.noHist:
            output = histogrammer(
                hist_collections=["QCD_negtag_lsf"],
                year=self._year,
                campaign=self._campaign,
            )

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
        # triggers = {trigger1 : [ptmin, ptmax], ...}
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
        # This has to be in ascending order, so that the prescale weight of the last passed trigger is kept

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
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level]

        for trigger in triggers:
            pruned_ev[trigger] = trigbools[trigger][event_level]

        if not isRealData:
            flav = ak.where(
                (event_jet.hadronFlavour == 0) & (event_jet.partonFlavour == 0),
                event_jet.hadronFlavour + 1,
                event_jet.hadronFlavour,
            )
            flav = ak.values_astype(
                ak.where(event_jet.partonFlavour == 21, flav + 6, flav), int
            )

        for tagger, tag_obj in btag_wp_dict[f"{self._year}_{self._campaign}"].items():
            for stringency, wp in tag_obj["b"].items():
                if stringency == "No":
                    continue

                negtag_mask = event_jet[f"btagNeg{tagger}B"] > wp
                negtag_jet = event_jet[negtag_mask][event_level]

                key = f"{tagger}{stringency}"
                pruned_ev[f"{key}_negtag_jet"] = negtag_jet

                if isRealData:
                    pruned_ev[f"{key}_negtag_jet", "flavor"] = ak.zeros_like(
                        negtag_jet.pt, dtype=int
                    )
                else:
                    pruned_ev[f"{key}_negtag_jet", "flavor"] = flav[negtag_mask][
                        event_level
                    ]

        if isRealData:
            pruned_ev["SelJet", "flavor"] = ak.zeros_like(
                event_jet[event_level].pt, dtype=int
            )
        else:
            pruned_ev["SelJet", "flavor"] = flav[event_level]

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
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev,
                output,
                weights,
                systematics,
                self.isSyst,
                self.SF_map,
                self._year,
                self._campaign,
            )
        # Output arrays
        if self.isArray:
            array_writer(
                self,
                pruned_ev,
                events,
                weights,
                systematics,
                dataset,
                isRealData,
                kinOnly=[],
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
