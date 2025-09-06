from coffea.util import load
import awkward as ak
import numpy as np
import sys, json, glob
import subprocess

def dump_lumi(output, fname):
    lumi, run = [], []
    for m in output.keys():
        for f in output[m].keys():
            lumi.extend(output[m][f]["lumi"].value)
            run.extend(output[m][f]["run"].value)

    # Sort runs and keep lumisections matched
    run, lumi = np.array(run), np.array(lumi)
    sorted_indices = np.lexsort((lumi, run))  # Sort by run first, then lumi
    run = run[sorted_indices]
    lumi = lumi[sorted_indices]
    # Create dictionary with ls values for each run
    dicts = {}
    for r in np.unique(run):
        dicts[str(r)] = lumi[run == r]
    # Convert to format for brilcalc
    for r in dicts.keys():
        ar = ak.singletons(ak.Array(dicts[r]))
        ars = ak.concatenate([ar, ar], axis=-1)
        dicts[r] = ak.values_astype(ars, int).tolist()

    with open(f"{fname}_lumi.json", "w") as outfile:
        json.dump(dicts, outfile, indent=2)

    brilcalc_cmd = (
        f"singularity -s exec --env PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages "
        f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest "
        f"brilcalc lumi --normtag src/BTVNanoCommissioning/data/lumiMasks/normtag_BRIL.json "
        f"-c web -i {fname}_lumi.json -u /pb &> {fname}_lumi.txt"
    )
    subprocess.run(brilcalc_cmd, shell=True)

    with open(f"{fname}_lumi.txt") as f:
        lumi_in_pb = f.read()
    rows = [line for line in lumi_in_pb.splitlines() if line.strip().startswith("|")]
    if not rows:
        raise RuntimeError("No summary table found in output")
    rows = [row for row in rows if "nfill" not in row]
    last_row = rows[-1]
    fields = [f.strip() for f in last_row.split("|") if f.strip()]
    totdelivered = float(fields[-2])
    totrecorded = float(fields[-1])
    print(f"Total delivered: {totdelivered:.3f} pb")
    print(f"Total recorded: {totrecorded:.3f} pb")

    
def dump_dataset(output, fname, alljson):
    jsonlist = glob.glob(alljson) if "*" in alljson else alljson.split(",")
    print("Original jsons:", jsonlist)
    original_list, list_from_coffea = {}, {}
    for j in jsonlist:
        old = json.load(open(j))
        for o in old.keys():
            if o not in original_list.keys():
                original_list[o] = []
            original_list[o].extend(old[o])

    for m in output.keys():
        for f in output[m].keys():
            if f not in list_from_coffea.keys():
                list_from_coffea[f] = list(output[m][f]["fname"])
            else:
                list_from_coffea[f] += list(set(output[m][f]["fname"]))
    failed = {}
    for t in original_list.keys():
        failed[t] = []
        if t not in list_from_coffea.keys():
            failed[t] = original_list[t]
            continue
        for f in original_list[t]:
            if f not in list_from_coffea[t]:
                failed[t].append(f)

    with open(f"{fname}_failed_dataset.json", "w") as outfile:
        json.dump(failed, outfile, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump procssed luminosity & failed coffea"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["all", "lumi", "failed"],
        help="Choose the function for dump luminosity(`lumi`)/failed files(`failed`) into json",
    )
    parser.add_argument(
        "-c",
        "--coffea",
        required=True,
        help="Processed coffea files, splitted by ,. Wildcard option * available as well.",
    )
    parser.add_argument(
        "-n", "--fname", required=True, help="Output name of jsons(with _lumi/_dataset)"
    )
    parser.add_argument(
        "-j",
        "--jsons",
        type=str,
        help="Original json files, splitted by ,. Wildcard option * available as well. ",
    )
    args = parser.parse_args()
    if len(args.coffea.split(",")) > 1:
        output = {i: load(i) for i in args.coffea.split(",")}
    elif "*" in args.coffea:
        args.coffea = glob.glob(args.coffea)
        output = {i: load(i) for i in args.coffea}
    else:
        output = {args.coffea: load(args.coffea)}

    if args.type == "all" or args.type == "lumi":
        print("===>Dump Processed Luminosity")
        dump_lumi(output, args.fname)
    if args.type == "all" or args.type == "failed":
        print("===>Dump Failed Files")
        dump_dataset(output, args.fname, args.jsons)
