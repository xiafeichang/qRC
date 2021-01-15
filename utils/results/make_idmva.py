from quantile_regression_chain import quantileRegression_chain_disc
from quantile_regression_chain import quantileRegression_chain
import pandas as pd
import numpy as np
import argparse
import yaml

import logging
logger = logging.getLogger("")

def setup_logging(level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(options):
    """
    Perform PhoIDMVA without going thg
    """

    stream = open(options.config,'r')
    inp=yaml.safe_load(stream)

    dataframes = inp['dataframes']
    showerShapes = inp['showerShapes']
    chIsos = inp['chIsos']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDirs = inp['weightsDirs']
    finalWeightsDirs = inp['finalWeightsDirs']

    data_file = '{}/{}'.format(workDir, dataframes['data'][options.EBEE]['input'])
    mc_file = '{}/{}'.format(workDir, dataframes['mc'][options.EBEE]['output'])

    variables = showerShapes + chIsos + ['probePhoIso']

    qrc = quantileRegression_chain(year, options.EBEE, workDir, variables)

    qrc.data = pd.read_hdf(data_file)
    qrc.MC = pd.read_hdf(mc_file)

    weights = ('/work/gallim/weights/id_mva/HggPhoId_94X_barrel_BDT_v2.weights.xml', '/work/gallim/weights/id_mva/HggPhoId_94X_endcap_BDT_v2.weights.xml')
    leg2016=False

    if options.EBEE == 'EB':
        qrc.data['probeScPreshowerEnergy'] = -999.*np.ones(qrc.data.index.size)
        qrc.MC['probeScPreshowerEnergy'] = -999.*np.ones(qrc.MC.index.size)
    elif options.EBEE == 'EE':
        qrc.data['probeScPreshowerEnergy'] = np.zeros(qrc.data.index.size)
        qrc.MC['probeScPreshowerEnergy'] = np.zeros(qrc.MC.index.size)

    if options.final:
        mvas = [("newPhoID","data",[]), ("newPhoIDcorrAll", "qr", qrc.vars), ("newPhoIDcorrAllFinal", "final", qrc.vars)]
    else:
        mvas = [("newPhoID","data",[]), ("newPhoIDcorrAll", "qr", qrc.vars)]

    qrc.computeIdMvas(mvas[:1],  weights,'data', n_jobs=options.n_jobs, leg2016=leg2016)
    qrc.computeIdMvas(mvas, weights,'mc', n_jobs=options.n_jobs , leg2016=leg2016)

    pref, suff = dataframes['mc'][options.EBEE]['output'].split('.')
    qrc.data.to_hdf('{}/{}'.format(workDir,dataframes['data'][options.EBEE]['output']),'df',mode='w',format='t')
    qrc.MC.to_hdf('{}/{}'.format(workDir, pref + '_with_MVA.' + suff),'df',mode='w',format='t')


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguments')
    requiredArgs.add_argument('-c','--config', action='store', default='quantile_config.yaml', type=str,required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-f','--final', action='store_true', default=False)
    optArgs.add_argument('-n','--n_jobs', action='store', type=int, default=1)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)
