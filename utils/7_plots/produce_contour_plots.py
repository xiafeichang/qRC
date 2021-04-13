import argparse
import pandas as pd
import mplhep as hep
import yaml
from itertools import combinations

from quantile_regression_chain.plotting.corr_plots import plot_contours

hep.set_style("CMS")



def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Plot variables distributions from an input file"
            )

    parser.add_argument(
            "--config",
            required=True,
            type=str,
            help="YAML file with the configurations for each variable"
            )

    parser.add_argument(
            "--output-dir",
            required=True,
            type=str,
            help="Output directory"
            )

    parser.add_argument(
            "--input-data",
            required=True,
            type=str,
            help="Pandas dataframe containing data values"
            )

    parser.add_argument(
            "--input-mc",
            required=True,
            type=str,
            help="Pandas dataframe containing mc values"
            )

    parser.add_argument(
            "--input-mc-corr",
            required=True,
            type=str,
            help="Pandas dataframe containing mc corrected values"
            )

    return parser.parse_args()


def main(args):
    data_file = args.input_data
    mc_file = args.input_mc
    mc_corr_file = args.input_mc_corr
    config_file = args.config
    output_dir = args.output_dir

    # Read config file
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    supp_cut = None
    if "cuts" in config_dict:
        supp_cut = config_dict["cuts"]

    conf_plots = config_dict["plots"]

    # Contour plots data
    data_df = pd.read_hdf(data_file)
    data_df = data_df.query(supp_cut)
    # Still stupid name conventions
    data_df['probePhiWidth'] = data_df['probePhiWidth_Sc']
    data_df['probeEtaWidth'] = data_df['probeEtaWidth_Sc']

    print("Producing contours for data")

    for x, y in combinations(conf_plots.keys(), 2):
        var_x = conf_plots[x]["var"]
        var_y = conf_plots[y]["var"]

        series_x = data_df[var_x]
        series_y = data_df[var_y]
        bins_array = [conf_plots[x]["bins"], conf_plots[y]["bins"]]
        ranges_array = [[conf_plots[x]["xmin"], conf_plots[x]["xmax"]], [conf_plots[y]["xmin"], conf_plots[y]["xmax"]]]
        fig_name = "data_{}_{}".format(x, y)
        plot_contours(fig_name, output_dir, series_x, series_y, bins_array, ranges_array)

    # Contour plots mc
    mc_df = pd.read_hdf(mc_file)
    mc_df = mc_df.query(supp_cut)

    print("Producing contours for mc")

    for x, y in combinations(conf_plots.keys(), 2):
        var_x = conf_plots[x]["var"]
        var_y = conf_plots[y]["var"]

        series_x = mc_df[var_x]
        series_y = mc_df[var_y]
        bins_array = [conf_plots[x]["bins"], conf_plots[y]["bins"]]
        ranges_array = [[conf_plots[x]["xmin"], conf_plots[x]["xmax"]], [conf_plots[y]["xmin"], conf_plots[y]["xmax"]]]
        fig_name = "mc_{}_{}".format(x, y)
        plot_contours(fig_name, output_dir, series_x, series_y, bins_array, ranges_array, mc_df['weight_clf'])

    # Contour plots mc corr
    mc_corr_df = pd.read_hdf(mc_corr_file)
    mc_corr_df = mc_corr_df.query(supp_cut)
    mc_corr_df['phoIdMVA_esEnovSCRawEn'] = mc_corr_df['phoIdMVA_esEnovSCRawEn_corr_1Reg']

    print("Producing contours for mc")

    for x, y in combinations(conf_plots.keys(), 2):
        var_x = conf_plots[x]["var"]
        var_y = conf_plots[y]["var"]

        series_x = mc_corr_df[var_x]
        series_y = mc_corr_df[var_y]
        bins_array = [conf_plots[x]["bins"], conf_plots[y]["bins"]]
        ranges_array = [[conf_plots[x]["xmin"], conf_plots[x]["xmax"]], [conf_plots[y]["xmin"], conf_plots[y]["xmax"]]]
        fig_name = "mc_corr_{}_{}".format(x, y)
        plot_contours(fig_name, output_dir, series_x, series_y, bins_array, ranges_array, mc_corr_df['weight_clf'])

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
