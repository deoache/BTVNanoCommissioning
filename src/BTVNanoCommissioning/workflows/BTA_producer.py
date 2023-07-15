import collections
import awkward as ak
import numpy as np
import uproot
from coffea import processor
from coffea.nanoevents.methods import vector
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.helpers.BTA_helper import (
    BTA_HLT,
    to_bitwise_trigger,
    get_hadron_mass,
    cumsum,
    is_from_GSP,
)
from BTVNanoCommissioning.helpers.func import update
from BTVNanoCommissioning.utils.correction import load_SF, JME_shifts


## Based on coffea_array_producer.ipynb from Congqiao
class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        isCorr=False,
        isJERC=True,
        isSyst=False,
        isArray=True,
        noHist=False,
        chunksize=75000,
    ):
        self._year = year
        self._campaign = campaign
        self.chunksize = chunksize
        self.isJERC = isJERC
        self.SF_map = load_SF(self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        events = missing_branch(events)
        shifts = []

        if "JME" in self.SF_map.keys() and self.isJERC:
            shifts = JME_shifts(
                shifts, self.SF_map, events, self._campaign, isRealData, False, True
            )
        else:
            shifts = [
                ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
            ]

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        events = missing_branch(events)
        # basic variables
        basic_vars = {
            "Run": events.run,
            "Evt": events.event,
            "LumiBlock": events.luminosityBlock,
            "rho": events.fixedGridRhoFastjetAll,
        }
        if not isRealData:
            basic_vars["nPU"] = events.Pileup.nPU
            basic_vars["nPUtrue"] = events.Pileup.nTrueInt
            basic_vars["pthat"] = events.Generator.binvar

        ###########
        #   HLT   #
        ###########
        triggers = BTA_HLT
        checkHLT = ak.Array([hasattr(events.HLT, _trig) for _trig in triggers])
        if ak.all(checkHLT == False):
            raise ValueError("HLT paths:", triggers, " are all invalid in", dataset)
        elif ak.any(checkHLT == False):
            print(
                np.array(triggers)[~checkHLT],
                " doesn't exist in",
                dataset,
                ", please check!",
            )

        pass_trig = np.array(
            [
                getattr(events.HLT, _trig, np.zeros(len(events), dtype="bool"))
                for _trig in triggers
            ]
        ).T

        basic_vars["BitTrigger"] = to_bitwise_trigger(
            pass_trig, ak.ArrayBuilder()
        ).snapshot()

        # PV
        PV = ak.zip(
            {
                "x": events.PV.x,
                "y": events.PV.y,
                "z": events.PV.z,
                "ndof": events.PV.ndof,
                "chi2": events.PV.chi2,
            }
        )
        if not isRealData:
            ###########
            # Quarks #
            ###########
            for quark_pdgid in [4, 5]:
                # finding b or c quarks
                quark_sel = (abs(events.GenPart.pdgId) == quark_pdgid) & (
                    (events.GenPart.statusFlags & (1 << 13)) > 0
                )  # is b/c-quark & is last copy
                quark_index = ak.local_index(events.GenPart.pdgId)[quark_sel]
                quarks = events.GenPart[quark_sel]
                out = ak.zip(
                    {
                        "pT": quarks.pt,
                        "eta": quarks.eta,
                        "phi": quarks.phi,
                        "pdgId": quarks.pdgId,
                        "status": quarks.status,
                        "fromGSP": is_from_GSP(quarks),
                    }
                )
                if quark_pdgid == 4:
                    cQuark = out
                elif quark_pdgid == 5:
                    bQuark = out

            ###########
            # Hadrons #
            ###########
            is_heavy_hadron = lambda p, pid: (abs(p.pdgId) // 100 == pid) | (
                abs(p.pdgId) // 1000 == pid
            )

            # finding D hadrons
            sel = is_heavy_hadron(events.GenPart, 4) & (
                (events.GenPart.statusFlags & (1 << 13)) > 0
            )  # PID match, is last copy

            chadrons = events.GenPart[sel]
            chadrons = chadrons[
                ~ak.any(is_heavy_hadron(chadrons.children, 4), axis=-1)
            ]  # should not select D hadron whose daughters also includes D hadrons
            DHadron = ak.zip(
                {
                    "pT": chadrons.pt,
                    "eta": chadrons.eta,
                    "phi": chadrons.phi,
                    "pdgID": chadrons.pdgId,
                    "mass": get_hadron_mass(chadrons.pdgId),
                    "muInDaughter": ak.any(abs(chadrons.children.pdgId) == 13, axis=-1),
                }
            )

            # finding B hadrons
            sel = is_heavy_hadron(events.GenPart, 5) & (
                (events.GenPart.statusFlags & (1 << 13)) > 0
            )  # PID match, is last copy
            bhadrons = events.GenPart[sel]
            BHadron = ak.zip(
                {
                    "pT": bhadrons.pt,
                    "eta": bhadrons.eta,
                    "phi": bhadrons.phi,
                    "pdgID": bhadrons.pdgId,
                    "mass": get_hadron_mass(bhadrons.pdgId),
                    "hasBdaughter": ak.values_astype(
                        ak.any(is_heavy_hadron(bhadrons.children, 5), axis=-1), int
                    ),  # B hadrons with B-daughters not removed
                }
            )

            ## find all daughter stable D hadrons and matched with D hadron list
            bhad_ch1G = bhadrons.children
            bhad_ch2G = bhadrons.children.children
            bhad_ch3G = bhadrons.children.children.children
            bhad_ch4G = bhadrons.children.children.children.children

            # for each B hadron: form a list of it daugthers which are:
            # D hadron & D does not include D hadron daugthers
            # i.e., find B -> D
            chad_asdau_1G = bhad_ch1G[
                is_heavy_hadron(bhad_ch1G, 4)
                & ~ak.any(is_heavy_hadron(bhad_ch2G, 4), axis=-1)
            ]
            # then find B -> D* -> D (store the final D)

            chad_asdau_2G = ak.flatten(
                bhad_ch2G[
                    is_heavy_hadron(bhad_ch2G, 4)
                    & ~ak.any(is_heavy_hadron(bhad_ch3G, 4), axis=-1)
                ],
                axis=3,
            )

            # then find B -> D** -> D* -> D (store the final D)
            chad_asdau_3G = ak.flatten(
                ak.flatten(
                    bhad_ch3G[
                        is_heavy_hadron(bhad_ch3G, 4)
                        & ~ak.any(is_heavy_hadron(bhad_ch4G, 4), axis=-1)
                    ],
                    axis=4,
                ),
                axis=3,
            )
            # combine all final D hadrons together
            # now in dim: (evt, bhad, chad_asdau)
            chad_asdau_final = ak.concatenate(
                [chad_asdau_1G, chad_asdau_2G, chad_asdau_3G], axis=2
            )
            chadrons_broadcast = ak.cartesian(
                [bhadrons, chadrons], axis=1, nested=True
            )[
                "1"
            ]  # broadcast to include bhad dimension: (evt, bhad, chad)

            # pair the D hadrons (as daughters) with the previous D hadron list
            pair = ak.cartesian(
                [chad_asdau_final, chadrons_broadcast], axis=2, nested=True
            )  # dim: (evt, bhad, chad_asdau, chad)
            chad_index = ak.argmax(
                pair["0"].pt == pair["1"].pt, axis=-1
            )  # dim: (evt, bhad, chad_asdau)

            # fill D hadron index (exclude those with hasBdaughter==False)
            BHadron["DHadron1"] = ak.fill_none(
                ak.where(
                    (ak.num(chad_index, axis=-1) > 0) & ~BHadron.hasBdaughter,
                    ak.pad_none(chad_index, 2, axis=2)[:, :, 0],
                    ak.zeros_like(BHadron.pT, dtype=int) - 1,
                ),
                -1,
            )
            BHadron["DHadron2"] = ak.fill_none(
                ak.where(
                    (ak.num(chad_index, axis=-1) > 1) & ~BHadron.hasBdaughter,
                    ak.pad_none(chad_index, 2, axis=2)[:, :, 1],
                    ak.zeros_like(BHadron.pT, dtype=int) - 1,
                ),
                -1,
            )

            ###############
            # Genlep & V0 #
            ###############
            ## Genlep

            _fix = lambda x: ak.fill_none(x, 0)
            is_lep = (
                lambda p: (abs(_fix(p.pdgId)) == 11)
                | (abs(_fix(p.pdgId)) == 13)
                | (abs(_fix(p.pdgId)) == 15)
            )
            is_WZ = lambda p: (abs(_fix(p.pdgId)) == 23) | (abs(_fix(p.pdgId)) == 24)
            is_heavy_hadron = lambda p, pid: (abs(_fix(p.pdgId)) // 100 == pid) | (
                abs(_fix(p.pdgId)) // 1000 == pid
            )

            sel = (
                is_lep(events.GenPart)
                & ((events.GenPart.statusFlags & (1 << 13)) > 0)
                & (events.GenPart.pt > 3.0)
            )  # requires pT > 3 GeV
            genlep = events.GenPart[sel]

            # trace parents up to 4 generations (from BTA code)
            genlep_pa1G = genlep.parent
            genlep_pa2G = genlep.parent.parent
            genlep_pa3G = genlep.parent.parent.parent
            genlep_pa4G = genlep.parent.parent.parent.parent
            istau = abs(genlep_pa1G.pdgId) == 15
            isWZ = is_WZ(genlep_pa1G) | is_WZ(genlep_pa2G)
            isD = is_heavy_hadron(genlep_pa1G, 4) | is_heavy_hadron(genlep_pa2G, 4)
            isB = (
                is_heavy_hadron(genlep_pa1G, 5)
                | is_heavy_hadron(genlep_pa2G, 5)
                | is_heavy_hadron(genlep_pa3G, 5)
                | is_heavy_hadron(genlep_pa4G, 5)
            )

            Genlep = ak.zip(
                {
                    "pT": genlep.pt,
                    "eta": genlep.eta,
                    "phi": genlep.phi,
                    "pdgID": genlep.pdgId,
                    "status": genlep.status,
                    "mother": ak.fill_none(
                        ak.where(isB | isD, 5 * isB + 4 * isD, 10 * istau + 100 * isWZ),
                        0,
                    ),
                }
            )

            # V0
            is_V0 = lambda p: (abs(p.pdgId) == 310) | (abs(p.pdgId) == 3122)
            sel = is_V0(events.GenPart) & ((events.GenPart.statusFlags & (1 << 13)) > 0)
            genV0 = events.GenPart[sel]

            # finding charged daughters with pT > 1
            is_lep = (
                lambda p: (abs(_fix(p.pdgId)) == 11)
                | (abs(_fix(p.pdgId)) == 13)
                | (abs(_fix(p.pdgId)) == 15)
            )
            is_charged_light_hadron = (
                lambda p: (abs(p.pdgId) == 211)
                | (abs(p.pdgId) == 213)
                | (abs(p.pdgId) == 321)
                | (abs(p.pdgId) == 323)
            )
            genV0_ch_charged = genV0.children[
                (is_charged_light_hadron(genV0.children) | is_lep(genV0.children))
                & (genV0.children.pt > 1.0)
            ]
            ncharged = ak.num(genV0_ch_charged, axis=2)

            # check acceptance (from the BTA code)
            svx = ak.fill_none(ak.firsts(genV0_ch_charged.vx, axis=-1), 0)
            svy = ak.fill_none(ak.firsts(genV0_ch_charged.vy, axis=-1), 0)
            svz = ak.fill_none(ak.firsts(genV0_ch_charged.vz, axis=-1), 0)
            radius = np.hypot(svx, svy)
            abseta = abs(genV0.eta)

            in_acceptance = (ncharged > 0) & (
                ((abseta < 2.0) & (radius < 7.3))
                | ((abseta < 2.5) & (radius < 4.4))
                | ((abseta > 1.8) & (abseta < 2.5) & (abs(svz) < 34.5))
            )

            genV0_inaccept = genV0[in_acceptance]
            GenV0 = ak.zip(
                {
                    "pT": genV0_inaccept.pt,
                    "eta": genV0_inaccept.eta,
                    "phi": genV0_inaccept.phi,
                    "pdgId": genV0_inaccept.pdgId,
                    "nCharged": ncharged[in_acceptance],
                    "SVx": svx[in_acceptance],
                    "SVy": svy[in_acceptance],
                    "SVz": svz[in_acceptance],
                }
            )

        ###############
        #     Jet     #
        ###############
        jet = events.Jet[
            (events.Jet.pt > 20.0) & (abs(events.Jet.eta) < 2.5)
        ]  # basic selection
        zeros = ak.zeros_like(jet.pt, dtype=int)
        Jet = ak.zip(
            {
                # basic kinematics
                "pT": jet.pt,
                "eta": jet.eta,
                "phi": jet.phi,
                "mass": jet.mass,
                "uncorrpt": jet.pt_raw,
                # jet ID/pileup ID // !!!
                "looseID": jet.jetId >= 2,
                "tightID": jet.jetId >= 4,
                "tightlepvetoID": jet.jetId >= 6,
                # pileup ID (essentially userInt('puId106XUL18Id') for UL18)
                # PU ID for Run 3 is not ready
                # 'pileup_tightID':ak.values_astype((jet.puId & (1 << 0) > 0) | (jet.pt > 50.), int)  ,
                # 'pileup_mediumID': ak.values_astype((jet.puId & (1 << 1) > 0) | (jet.pt > 50.), int) ,
                # 'pileup_looseID': ak.values_astype((jet.puId & (1 << 2) > 0) | (jet.pt > 50.), int) ,
                # taggers/vars // JP/JBP to be calibrated.. !!!
                "area": jet.area,
                "Proba": jet.Proba,
                "ProbaN": jet.ProbaN,
                "Bprob": jet.Bprob,
                "BprobN": jet.BprobN,
                # Deep Jet
                "DeepFlavourBDisc": jet.btagDeepFlavB,
                "DeepFlavourCvsLDisc": jet.btagDeepFlavCvL,
                "DeepFlavourCvsBDisc": jet.btagDeepFlavCvB,
                "DeepFlavourBDisc_b": jet.btagDeepFlavB_b,
                "DeepFlavourBDisc_bb": jet.btagDeepFlavB_bb,
                "DeepFlavourBDisc_lepb": jet.btagDeepFlavB_lepb,
                "DeepFlavourCDisc": jet.btagDeepFlavC,
                "DeepFlavourUDSDisc": jet.btagDeepFlavUDS,
                "DeepFlavourGDisc": jet.btagDeepFlavG,
                "DeepFlavourBDiscN": jet.btagNegDeepFlavB,
                "DeepFlavourCvsLDiscN": jet.btagNegDeepFlavCvL,
                "DeepFlavourCvsBDiscN": jet.btagNegDeepFlavCvB,
                "DeepFlavourBDisc_bN": jet.btagNegDeepFlavB_b,
                "DeepFlavourBDisc_bbN": jet.btagNegDeepFlavB_bb,
                "DeepFlavourBDisc_lepbN": jet.btagNegDeepFlavB_lepb,
                "DeepFlavourQGDiscN": jet.btagNegDeepFlavQG,
                "DeepFlavourCDiscN": jet.btagNegDeepFlavC,
                "DeepFlavourUDSDiscN": jet.btagNegDeepFlavUDS,
                "DeepFlavourGDiscN": jet.btagNegDeepFlavG,
                # Particle net
                "PNetBDisc": jet.btagPNetB,
                "PNetCvsLDisc": jet.btagPNetCvL,
                "PNetCvsBDisc": jet.btagPNetCvB,
                "PNetQvsGDisc": jet.btagPNetQvG,
                "PNetTauvsJetDisc": jet.btagPNetTauVJet,
                "PNetRegPtRawCorr": jet.PNetRegPtRawCorr,
                "PNetRegPtRawCorrNeutrino": jet.PNetRegPtRawCorrNeutrino,
                "PNetRegPtRawRes": jet.PNetRegPtRawRes,
                "PNetBDisc_b": jet.btagPNetProbB,
                "PNetCDisc": jet.btagPNetProbC,
                "PNetUDSDisc": jet.btagPNetProbUDS,
                "PNetGDisc": jet.btagPNetProbG,
                "PNetBDiscN": jet.btagNegPNetB,
                "PNetCvsLDiscN": jet.btagNegPNetCvL,
                "PNetCvsBDiscN": jet.btagNegPNetCvB,
                "PNetBDisc_bN": jet.btagNegPNetProbB,
                "PNetCDiscN": jet.btagNegPNetProbC,
                "PNetUDSDiscN": jet.btagNegPNetProbUDS,
                "PNetGDiscN": jet.btagNegPNetProbG,
                # ParT
                "ParTBDisc": jet.btagRobustParTAK4B,
                "ParTCvsLDisc": jet.btagRobustParTAK4CvL,
                "ParTCvsBDisc": jet.btagRobustParTAK4CvB,
                "ParTQvsGDisc": jet.btagRobustParTAK4QG,
                "ParTBDisc_b": jet.btagRobustParTAK4B_b,
                "ParTBDisc_bb": jet.btagRobustParTAK4B_bb,
                "ParTBDisc_lepb": jet.btagRobustParTAK4B_lepb,
                "ParTCDisc": jet.btagRobustParTAK4C,
                "ParTUDSDisc": jet.btagRobustParTAK4UDS,
                "ParTGDisc": jet.btagRobustParTAK4G,
                "ParTBDiscN": jet.btagNegRobustParTAK4B,
                "ParTCvsLDiscN": jet.btagNegRobustParTAK4CvL,
                "ParTCvsBDiscN": jet.btagNegRobustParTAK4CvB,
                "ParTQvsGDiscN": jet.btagNegRobustParTAK4QG,
                "ParTBDisc_bN": jet.btagNegRobustParTAK4B_b,
                "ParTBDisc_bbN": jet.btagNegRobustParTAK4B_bb,
                "ParTBDisc_lepbN": jet.btagNegRobustParTAK4B_lepb,
                "ParTCDiscN": jet.btagNegRobustParTAK4C,
                "ParTUDSDiscN": jet.btagNegRobustParTAK4UDS,
                "ParTGDiscN": jet.btagNegRobustParTAK4G,
            }
        )
        if isRealData:
            Jet["vetomap"] = jet.veto

        if not isRealData:
            Jet["nbHadrons"] = jet.nBHadrons
            Jet["ncHadrons"] = jet.nCHadrons
            Jet["partonFlavour"] = jet.partonFlavour
            Jet["hadronFlavour"] = jet.hadronFlavour
            # corrected flavours defined in BTA
            Jet["flavour"] = ak.where(
                jet.hadronFlavour != 0,
                jet.hadronFlavour,
                ak.where(
                    (abs(jet.partonFlavour) == 4) | (abs(jet.partonFlavour) == 5),
                    zeros,
                    jet.partonFlavour,
                ),
            )

            # genJet pT
            genJetIdx = ak.where(
                jet.genJetIdx < ak.num(events.GenJet), jet.genJetIdx, zeros - 1
            )  # in case the genJet index out of range
            Jet["genpt"] = ak.where(
                genJetIdx >= 0, events.GenJet[genJetIdx].pt, zeros - 1
            )

            # gen-level jet cleaning aginst prompt leptons
            genlep_prompt = genlep[(Genlep.mother != 0) & (Genlep.mother % 10 == 0)]
            overlapped = ak.zeros_like(jet.pt, dtype=bool)

            for lep_pid, dr_thres in [(11, 0.2), (13, 0.2), (15, 0.3)]:
                lep = genlep_prompt[abs(genlep_prompt.pdgId) == lep_pid]
                pair = ak.cartesian(
                    [jet, lep], axis=1, nested=True
                )  # dim: (event, jet, lep)
                dr_min = ak.min(
                    pair["0"].delta_r(pair["1"]), axis=-1
                )  # dim: (event, jet)
                dr_min = ak.fill_none(dr_min, 99.0)  # for 0 lepton case, assign dr=99
                overlapped = overlapped | (dr_min < dr_thres)
            Jet["flavourCleaned"] = ak.where(
                overlapped, zeros - 999, Jet["flavour"]
            )  # fill -999 for jets with lepton overlapping

            # jet cleaning aginst pileup
            Jet["flavourCleaned"] = ak.where(
                Jet["genpt"] < 8.0, zeros, Jet["flavourCleaned"]
            )

        # CSV inputs per jets: retreived from the DeepCSV input given they are the same
        TagVarCSV = ak.zip(
            {
                "trackJetPt": jet.DeepCSV_trackJetPt,
                "vertexCategory": jet.DeepCSV_vertexCategory,
                "jetNSecondaryVertices": jet.DeepCSV_jetNSecondaryVertices,
                "trackSumJetEtRatio": jet.DeepCSV_trackSumJetEtRatio,
                "trackSumJetDeltaR": jet.DeepCSV_trackSumJetDeltaR,
                "vertexMass": jet.DeepCSV_vertexMass,
                "vertexNTracks": jet.DeepCSV_vertexNTracks,
                "vertexEnergyRatio": jet.DeepCSV_vertexEnergyRatio,
                "vertexJetDeltaR": jet.DeepCSV_vertexJetDeltaR,
            }
        )

        ###############
        #    Trkj     #
        ###############
        if "QCD" in events.metadata["dataset"]:
            trkj = events.JetPFCands[
                (events.JetPFCands.pf.trkQuality != 0) & (events.JetPFCands.pt > 1.0)
            ]

            # selection for "TrkInc" in BTA
            trkj = trkj[
                (trkj.pf.trkHighPurity == 1)
                & (trkj.pf.trkAlgo != 9)
                & (trkj.pf.trkAlgo != 10)
                & (trkj.pt > 5.0)
                & (trkj.pf.numberOfHits >= 11)
                & (trkj.pf.numberOfPixelHits >= 2)
                & (trkj.pf.trkChi2 < 10)
                & (trkj.pf.lostOuterHits <= 2)
                & (trkj.pf.dz < 1.0)
            ]

            pair = ak.cartesian(
                [jet, trkj], axis=1, nested=True
            )  # dim: (event, jet, pfcand)
            matched = (pair["0"].pt_orig == pair["1"].jet.pt) & (
                pair["0"].delta_r(pair["1"].pf) < 0.4
            )
            trkj_jetbased = pair["1"][matched]  # dim: (event, jet, pfcand)

            # calculate pTrel
            vec = ak.zip(
                {
                    "pt": trkj_jetbased.pf.trkPt,
                    "eta": trkj_jetbased.pf.trkEta,
                    "phi": trkj_jetbased.pf.trkPhi,
                    "mass": trkj_jetbased.pf.mass,
                },
                behavior=vector.behavior,
                with_name="PtEtaPhiMLorentzVector",
            )
            ptperp = vec.dot(jet) / jet.p
            trkj_jetbased["ptrel"] = np.sqrt(np.maximum(vec.p**2 - ptperp**2, 0))

            # flatten jet-based track arrays
            trkj_jetbased_flat = ak.flatten(
                trkj_jetbased, axis=2
            )  # dim: (event, pfcand)
            trkj_jetbased_num = ak.num(trkj_jetbased, axis=2)

            TrkInc = ak.zip(
                {
                    "pt": ak.fill_none(trkj_jetbased_flat.pf.trkPt, 0),
                    "eta": ak.fill_none(trkj_jetbased_flat.pf.trkEta, 0),
                    "phi": ak.fill_none(trkj_jetbased_flat.pf.trkPhi, 0),
                    "ptrel": ak.fill_none(trkj_jetbased_flat.ptrel, 0),
                }
            )

            # assign the first and last index of matched muons to Jet
            lastidx = cumsum(trkj_jetbased_num)

            firstidx = ak.where(
                ak.num(trkj_jetbased_num) > 0,
                ak.concatenate([0, lastidx[:, :-1]], axis=1),
                lastidx,
            )

            Jet["nFirstTrkInc"] = firstidx
            Jet["nLastTrkInc"] = lastidx

            ###############
            #    PFMuon   #
            ###############

            # find muon PFCands inside a jet
            # based on SoftPFMuonTagInfoProducer:
            # https://github.com/cms-sw/cmssw/blob/10_6_X/RecoBTag/SoftLepton/plugins/SoftPFMuonTagInfoProducer.cc

            mutrkj = events.JetPFCands[
                (events.JetPFCands.pf.trkQuality != 0)
                & (events.JetPFCands.pt > 2.0)
                & (abs(events.JetPFCands.pf.pdgId) == 13)
            ]

            # match muon PFCands to muons
            pair = ak.cartesian(
                [mutrkj, events.Muon], axis=1, nested=True
            )  # dim: (event, pfcand, mu)
            matched = (
                (pair["0"].pf.pdgId == pair["1"].pdgId)
                & (pair["0"].pf.delta_r(pair["1"]) < 0.01)
                & ((pair["0"].pt - pair["1"].pt) / pair["0"].pt < 0.1)
            )
            mutrkj["mu"] = ak.firsts(
                pair["1"][matched], axis=2
            )  # dim same with mutrkj: (event, matched-pfcand), None if not matching with a muon

            # match muon PFCands also to jets
            pair = ak.cartesian(
                [jet, mutrkj], axis=1, nested=True
            )  # dim: (event, jet, pfcand)
            matched = pair["0"].pt_orig == pair["1"].jet.pt
            mutrkj_jetbased = pair["1"][matched]  # dim: (event, jet, matched-pfcand)

            # further requirement: must match with a muon and require the loose ID
            mutrkj_jetbased = mutrkj_jetbased[
                ak.fill_none(mutrkj_jetbased.mu.looseId, False)
            ]  # dim: (event, jet, matched-muons)
            mu_jetbased = mutrkj_jetbased.mu

            # calculate pTrel and other kinematics
            ptperp = mu_jetbased.dot(jet) / jet.p
            mu_jetbased["ptrel"] = np.sqrt(mu_jetbased.p**2 - ptperp**2)
            # ratio and ratioRel seems different from the BTA value..
            mu_jetbased["ratio"] = mu_jetbased.pt / jet.pt
            mu_jetbased["ratioRel"] = (
                mu_jetbased.t * jet.t - mu_jetbased.dot(jet)
            ) / jet.mass2

            mu_jetbased["deltaR"] = mu_jetbased.delta_r(jet)

            # quality
            zeros = ak.zeros_like(mu_jetbased.pt, dtype=int)
            mu_jetbased["GoodQuality"] = zeros
            mu_jetbased["GoodQuality"] = ak.where(
                mu_jetbased.isGlobal, zeros + 1, mu_jetbased["GoodQuality"]
            )
            mu_jetbased["GoodQuality"] = ak.where(
                (mu_jetbased["GoodQuality"] == 1)
                & (mu_jetbased.nStations >= 2)
                & (mutrkj_jetbased.pf.numberOfHits >= 11)
                & (mutrkj_jetbased.pf.numberOfPixelHits >= 2)
                & (mutrkj_jetbased.pf.trkChi2 < 10),
                zeros + 2,
                mu_jetbased["GoodQuality"],
            )

            # flatten jet-based track arrays
            mu_jetbased_flat = ak.flatten(mu_jetbased, axis=2)  # dim: (event, pfcand)
            mu_jetbased_jetidx_flat = ak.flatten(
                ak.broadcast_arrays(ak.local_index(jet, axis=1), mu_jetbased.pt)[0],
                axis=2,
            )
            mu_jetbased_num = ak.num(mu_jetbased, axis=2)

            PFMuon = ak.zip(
                {
                    "IdxJet": ak.fill_none(mu_jetbased_jetidx_flat, 0),
                    "pt": ak.fill_none(mu_jetbased_flat.pt, 0),
                    "eta": ak.fill_none(mu_jetbased_flat.eta, 0),
                    "phi": ak.fill_none(mu_jetbased_flat.phi, 0),
                    "ptrel": ak.fill_none(mu_jetbased_flat.ptrel, 0),
                    "ratio": ak.fill_none(mu_jetbased_flat.ratio, 0),
                    "ratioRel": ak.fill_none(mu_jetbased_flat.ratioRel, 0),
                    "deltaR": ak.fill_none(mu_jetbased_flat.deltaR, 0),
                    "IP": ak.fill_none(
                        mu_jetbased_flat.ip3d
                        * np.sign(mu_jetbased_flat.dxy + mu_jetbased_flat.dz),
                        0,
                    ),
                    "IPsig": ak.fill_none(
                        mu_jetbased_flat.sip3d
                        * np.sign(mu_jetbased_flat.dxy + mu_jetbased_flat.dz),
                        0,
                    ),
                    "IP2D": ak.fill_none(mu_jetbased_flat.dxy, 0),
                    "IP2Dsig": ak.fill_none(
                        mu_jetbased_flat.dxy / mu_jetbased_flat.dxyErr, 0
                    ),
                    "GoodQuality": ak.fill_none(mu_jetbased_flat.GoodQuality, 0),
                }
            )

            # assign the first and last index of matched muons to Jet
            lastidx = cumsum(mu_jetbased_num)
            firstidx = ak.where(
                ak.num(mu_jetbased_num) > 0,
                ak.concatenate([0, lastidx[:, :-1]], axis=1),
                lastidx,
            )
            Jet["nFirstSM"] = firstidx
            Jet["nLastSM"] = lastidx

        ###############
        #  Write root #
        ###############

        output = {
            **basic_vars,
            "PV": PV,
            "Jet": Jet,
            "TagVarCSV": TagVarCSV,
        }
        if "QCD" in events.metadata["dataset"]:
            output["TrkInc"] = TrkInc
            output["PFMuon"] = PFMuon
        if not isRealData:
            output["bQuark"] = bQuark
            output["cQuark"] = cQuark
            output["BHadron"] = BHadron
            output["DHadron"] = DHadron
            output["Genlep"] = Genlep
            output["GenV0"] = GenV0
        fname = f"{dataset}_f{events.metadata['filename'].split('_')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"
        with uproot.recreate(fname) as fout:
            output_root = {}
            for bname in output.keys():
                if not output[bname].fields:
                    output_root[bname] = ak.packed(ak.without_parameters(output[bname]))
                else:
                    b_nest = {}
                    for n in output[bname].fields:
                        b_nest[n] = ak.packed(ak.without_parameters(output[bname][n]))
                    output_root[bname] = ak.zip(b_nest)
            fout["btagana/ttree"] = output_root

        return {dataset: len(events)}

    def postprocess(self, accumulator):
        return accumulator
