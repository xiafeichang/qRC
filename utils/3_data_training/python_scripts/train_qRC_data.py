import argparse
import yaml
import os
from quantile_regression_chain.quantileRegression_chain import quantileRegression_chain as QReg_C

import logging
logger = logging.getLogger("")

def setup_logging(level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(options):

    stream = open(options.config,'r')
    inp=yaml.safe_load(stream)

    df_name = inp['dataframes']['data_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if options.split is not None:
        print ('Using split dfs for training two sets of weights!')
        df_name = df_name.replace('.h5', '') + '_spl{}.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    if year == '2017' or year == '2018':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIeIp']
    elif year == '2016':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIetaIphi']

    num_hidden_layers = 5
    num_units = [500, 300, 200, 100, 50]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']

    qRC = QReg_C(year,options.EBEE,workDir,variables)
    qRC.loadDataDF(df_name,0,options.n_evts,rsh=False,columns=columns)

    scale_file = '{}/scale_par_{}.h5'.format(weightsDir, options.EBEE)
    if not os.path.exists(scale_file):
        qRC.gen_scale_par(weightsDir, 'scale_par_{}.h5'.format(options.EBEE))
    else: 
        qRC.scale_par = scale_file

    qRC.trainOnData(options.variable,num_hidden_layers,num_units,act,weightsDir=weightsDir,clusterconfig=options.clusterconfig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-i','--clusterconfig', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options=parser.parse_args()
    main(options)

