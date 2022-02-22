#!/bin/bash

EBEE=$1

for var in "probeCovarianceIeIp" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeR9" "probeS4";
do
    echo var: ${var}
    python $PWD/python_scripts/train_qRC_data.py -c config/config_qRC_training_5M.yaml -i config/dask_cluster_config.yml -v ${var} -N 4700000 -E ${EBEE}
done

