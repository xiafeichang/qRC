import argparse

from dask.distributed import Client, LocalCluster, progress, wait, get_client

from quantile_regression_chain import QRCScheduler

import logging
logger = logging.getLogger("")

def parse_arguments():
    parser = argparse.ArgumentParser(
            description = '')

    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        type=str,
        help="Path to YAML config file")

    parser.add_argument(
        "-sc",
        "--slurm_config",
        type=str,
        help="Path to YAML config file with information to setup a SLURM cluster")

    parser.add_argument(
        "-cl",
        "--cluster_id",
        type=str,
        help="")

    return parser.parse_args()

def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def main(args):
    # Parse cmd line arguments
    config_file = args.config_file
    slurm_config = args.slurm_config
    cluster_id = args.cluster_id

    qrc_scheduler = QRCScheduler(config_file)

    if slurm_config:
        qrc_scheduler.setup_slurm_cluster(slurm_config)
    elif cluster_id:
        qrc_scheduler.connect_to_cluster(cluster_id)
    else:
        qrc_scheduler.client = None

    qrc_scheduler.load_dataframes()

    qrc_scheduler.train_regressors()

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging('train_all_with_scheduler.log', logging.INFO)
    #setup_logging('train_all_with_scheduler.log')
    main(args)
