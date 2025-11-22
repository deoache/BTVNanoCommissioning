import hist
from BTVNanoCommissioning.utils.selection import btag_wp_dict


def get_histograms(axes, **kwargs):
    year = kwargs.get("year", None)
    if year == None:
        raise ValueError(
            "year is not specified. Please specify the year in histogrammer."
        )
    campaign = kwargs.get("campaign", None)
    if campaign == None:
        raise ValueError(
            "campaign is not specified. Please specify the campaign in histogrammer."
        )

    hists = {}
    hists["qcd_jet_pt"] = hist.Hist(
        axes["syst"],
        axes["flav"],
        axes["jpt"],
        hist.storage.Weight(),
    )
    for tagger, tag_obj in btag_wp_dict[f"{year}_{campaign}"].items():
        for stringency, wp in tag_obj["b"].items():
            if stringency == "No":
                continue
            key = f"{tagger}{stringency}"
            hists[f"{key}_qcd_negtag_jet_pt"] = hist.Hist(
                axes["syst"],
                axes["flav"],
                axes["jpt"],
                hist.storage.Weight(),
            )
    return hists
