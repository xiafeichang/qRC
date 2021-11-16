#!/bin/bash

EBEE=$1

for var in "probeCovarianceIeIp" "probeSigmaIeIe" "probeEtaWidth_Sc" "probePhiWidth_Sc" "probeR9" "probeS4";
do
    echo var: ${var}
    python $PWD/python_scripts/train_qRC_data.py -c config/config_qRC_training_5M.yaml -v ${var} -q 0.5 -N 100 -E ${EBEE}
done

