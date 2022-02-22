import argparse
import yaml
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

    df_name_data = inp['dataframes']['data_{}'.format(options.EBEE)]
    df_name_mc = inp['dataframes']['mc_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if options.split is not None:
        print('Using split dfs for training two sets of weights!')
        df_name_data = df_name_data.replace('.h5', '') + '_spl{}.h5'.format(options.split)
        df_name_mc = df_name_mc.replace('.h5', '') + '_spl{}.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    num_hidden_layers = 5
    num_units = [500, 300, 200, 100, 50]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']


    qRC = QReg_C(year,options.EBEE,workDir,variables)
    qRC.loadMCDF(df_name_mc,0,options.n_evts,rsh=False)
    qRC.loadDataDF(df_name_data,0,options.n_evts,rsh=False)

    qRC.scale_par = '{}/scale_par_{}.h5'.format(weightsDir, options.EBEE)
    qRC.trainAllMC(num_hidden_layers,num_units,act,weightsDir=weightsDir,clusterconfig=options.clusterconfig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-i','--clusterconfig', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)
