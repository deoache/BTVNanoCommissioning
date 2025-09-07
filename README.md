
# BTVNanoCommissioning
[![Linting](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml)
[![btag ttbar](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml)
[![ctag ttbar](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_ttbar_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_ttbar_workflow.yml)
[![ctag DY+jets Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_DY_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_DY_workflow.yml)
[![ctag W+c Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_Wc_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_Wc_workflow.yml)
[![BTA Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/BTA_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/BTA_workflow.yml)
[![QCD Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/QCD_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/QCD_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repository for Commissioning studies in the BTV POG based on (custom) nanoAOD samples


This framework is based on [coffea](https://coffeateam.github.io/coffea/) and using [btvnano](https://btv-wiki.docs.cern.ch/SoftwareAlgorithms/PFNano/) as input. The framework is also used as frontend for the btv automation task [autobtv](https://gitlab.cern.ch/cms-analysis/btv/software-and-algorithms/autobtv)

This framework is based on [coffea processor](https://coffeateam.github.io/coffea/concepts.html#coffea-processor). Each workflow can be a separate **processor** file in the `workflows`, creating the mapping from `PFNano` to the histograms as `coffea` file or creating `.root` files by saving awkward arrays. Workflow processors can be passed to the `runner.py` script along with the fileset these should run over. Multiple executors can be chosen
(`iterative` - one by one, `futures` - multiprocessing). Scale out to clusters depend on facilities. Obtain the histograms as plot(`.pdf`) or save to template `.root` file with dedicated scripts

The minimum requirement commands are shown in follow, specified the selections, datataset, campaign and year
```
python runner.py --workflow ttsemilep_sf --json metadata/test_bta_run3.json --campaign Summer22EERun3 --year 2022
```
- Detailed documentation [here](https://btvnanocommissioning.readthedocs.io/en/latest/)
- To running the commissioning task or producing the template: go to [Preparation for commissioning/SFs tasks](https://btvnanocommissioning.readthedocs.io/en/latest/user.html)
- To develop new workflow, the instruction can be found in [Instruction for developers](https://btvnanocommissioning.readthedocs.io/en/latest/user.html)
- Current working in progress [issues](https://gitlab.cern.ch/cms-btv-coordination/tasks/-/issues/?label_name%5B%5D=Software%3A%3A%20BTVnano%20%26CommFW)



## Setup 

You can install your [standalone conda envrionment](#standalone-conda-environment) via `yaml` or on the lxplus you can directly jump to [setup](#setup-the-framework)
### Standalone conda environment
> [!Caution]
> suggested to install under `bash` environment


For installing Micromamba, see [[here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)]
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# Run and follow instructions on screen
bash Miniforge3-$(uname)-$(uname -m).sh

micromamba activate
```
NOTE: always make sure that conda, python, and pip point to local micromamba installation (`which conda` etc.).


You can simply create the environment through the existing `test_env.yml` under your micromamba environment using micromamba, and activate it
```
micromamba env create -f test_env.yml 

```
### Setup the framework

```bash
# activate enviroment once you have coffea framework 
conda/micromamba activate  /eos/home-m/milee/miniforge3/envs/btv_coffea

# Once the environment is set up:
git clone https://github.com/deoache/BTVNanoCommissioning.git
pip install -e .
pip install -e .[dev, doc] # for developer
```

You can still install additional packages itself by `pip install $PACKAGE`

Make sure that 'dasgoclient' command is found by ensuring it is in your `$PATH` variable.
Please, run `source env_setup_complete.sh` after having activated the environment to make sure that the `$PATH` variable is correctly configured.


## QCD light workfow

### 1. Workfow

#### Output Histograms

Each histogram is organized along three independent axes: Systematic axis (`syst`), Flavor axis (`flav`) and Jet transverse momentum axis (`jpt`). Based on these axes, the workflow defines several histograms:

- **Inclusive jet histogram** (`jet_pt`): All selected jets as a function of $p_T$ and flavor, with systematic variations tracked along the additional axis.

- **Post-tag histograms** (`UParTAK4B_{wp}_postag_jet_pt`): for each working point of the UParT b-tagging algorithm (Loose, Medium, Tight), a histogram is created containing jets that pass the corresponding discriminator threshold.

- **Neg-tag histograms** (`UParTAK4B_{wp}_negtag_jet_pt`): these histograms store jets that pass the corresponding discriminator threshold for the negative UParT b-tagging discriminator


#### Triggers and prescale weights

The analysis uses PFJet triggers, each associated with a specific $p_T$ range for the leading jet. The selection requires that the leading jet falls within the range corresponding to a given trigger:
```python
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
```
For each event, `req_trig` is constructed to be true if at least one trigger fires and the leading jet $p_T$ lies within the required range:
```python
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
```

Triggers are subject to prescale factors, which reduce the accepted event rate. To correct for this, prescale weights (psweight) are applied
```python
weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
if isRealData:
    run_num = "378985_386951" # 2024
    psweight = np.zeros(len(pruned_ev))
    for trigger, trigbool in trigbools.items():
        psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trigger}_run{run_num}.json"
        pseval = correctionlib.CorrectionSet.from_file(psfile)
        thispsweight = pseval["prescaleWeight"].evaluate(
            pruned_ev.run,
            f"HLT_{trigger}",
            ak.values_astype(pruned_ev.luminosityBlock, np.float32),
        )
        psweight = ak.where(trigbool[event_level], thispsweight, psweight)
    weights.add("psweight", psweight)

```
To build the prescale weights use the `dump_prescale.py` script:
```bash
# build prescale weights for HLT_PFJet40
python scripts/dump_prescale.py -v --HLT PFJet40 --lumimask <golden_json>
```
For 2024 use `src/BTVNanoCommissioning/data/lumiMasks/Cert_Collisions2024_378981_386951_Golden.json` as golden json.


### 2. Input filesets

To generate input fileset type:

```bash
# MC
python scripts/fetch.py -c Summer24 -i metadata/Summer24/QCD_light_sf_mc -o MC_Summer24_2024_QCD_light_sf.json --skipvalidation
```
```bash
# Data
python scripts/fetch.py -c Summer24 -i metadata/Summer24/QCD_light_sf_data -o data_Summer24_2024_QCD_light_sf.json --skipvalidation
```

### 4. Submit jobs to Condor

You can submit jobs to condor (at lxplus) with:

```bash
# MC
python condor_lxplus/submitter.py \
  --workflow QCD_light_sf \
  --json metadata/Summer24/MC_Summer24_2024_QCD_light_sf.json \
  --campaign Summer24 \
  --year 2024 \
  --skipbadfiles \
  --jobName qcd_light_sf_mc \
  --outputDir <mc_outputDir> \
  --condorFileSize 10 \
  --submit

# DATA
python condor_lxplus/submitter.py \
  --workflow QCD_light_sf \
  --json metadata/Summer24/data_Summer24_2024_QCD_light_sf.json \
  --campaign Summer24 \
  --year 2024 \
  --skipbadfiles \
  --jobName qcd_light_sf_data \
  --outputDir <data_outputDir> \
  --condorFileSize 10 \
  --submit
```

To check for missing files (add `-u` to update condor file):

```bash
# MC
python scripts/missingFiles.py -j qcd_light_sf_mc -o <mc_outputDir>

# DATA
python scripts/missingFiles.py -j qcd_light_sf_data -o <data_outputDir>
```

### 5. Luminosity computation

To extract the integrated luminosity from the output coffea files
```bash
python scripts/dump_processed.py -t lumi -c '<data_outputDir>/*/*.coffea' -n 2024 -j metadata/Summer24/data_Summer24_2024_QCD_light_sf.json
```


### 6. Data/MC plots

To produce Data/MC comparison plots:
```bash
python scripts/plotdataMC.py \
  -i '<data_outputDir>/*/*.coffea,<mc_outputDir>/*/*.coffea' \
  --lumi 106356.676392530 \
  -p QCD_light_sf \
  -v all \
  --split flavor \
  --log \
  --xrange 0,1000
```