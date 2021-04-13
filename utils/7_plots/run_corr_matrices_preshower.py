import argparse
import pandas as pd
import numpy as np
import warnings
from quantile_regression_chain.plotting import corr_plots
from quantile_regression_chain.syst import qRC_systematics as syst
import os
import uproot

def main(options):
    # Setup
    # Open test dataframes with all variables
    original_data = pd.read_hdf('/work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_sc/df_data_EE_test.h5')
    original_mc = pd.read_hdf('/work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_sc/df_mc_EE_test.h5')

    # Define variables
    base_vars = ['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'mass'] #pt already contained in diff var

    variables = ["probePt","probeScEta","probePhi",'rho','probeScEnergy','probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth','probePhoIso03','probeChIso03','probeChIso03worst', 'phoIdMVA_esEnovSCRawEn']
    variables_corr = ["probePt","probeScEta","probePhi",'rho','probeScEnergy','probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth','probePhoIso03','probeChIso03','probeChIso03worst', 'phoIdMVA_esEnovSCRawEn_corr_1Reg']

    varrs      = variables+base_vars+variables_corr
    varrs_data = variables+base_vars

    # Define cut
    if options.EBEE=='EE':
        cut_string = 'abs(probeScEta)>1.56 and tagPt>40 and probePt>25 and mass>80 and mass<100 and probePassEleVeto==0 and abs(tagScEta)<2.5 and abs(probeScEta)<2.5 and probeSigmaIeIe<0.028'
    else:
        raise NameError('Region has to be EE')

    df_mc = pd.read_hdf(options.mc)
    df_data = pd.read_hdf(options.data)

    # Changing stupid naming convention
    #df_mc['probePhiWidth'] = original_mc['probePhiWidth']
    #df_mc['probeEtaWidth'] = original_mc['probeEtaWidth']
    #df_data['probePhiWidth'] = df_data['probePhiWidth_Sc']
    #df_data['probeEtaWidth'] = df_data['probeEtaWidth_Sc']
    df_mc['probePhoIso03'] = original_mc['probePhoIso03']
    #df_data['probePhoIso03'] = original_data['probePhoIso03']

    df_mc = df_mc.query(cut_string)
    df_data = df_data.query(cut_string)

    corrMatspt = []
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt<35'),df_data.query('probePt<35'),variables,variables_corr,'weight_clf','{} qRC probePt<35'.format(options.EBEE)))
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt>35 and probePt<50'),df_data.query('probePt>35 and probePt<50'),variables,variables_corr,'weight_clf','{} qRC 35 < probePt < 50'.format(options.EBEE)))
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt>50'),df_data.query('probePt>50'),variables,variables_corr,'weight_clf','{} qRC probePt > 50'.format(options.EBEE)))
    for corrMat in corrMatspt:
        corrMat.plot_corr_mat('mc')
        corrMat.plot_corr_mat('mcc')
        corrMat.plot_corr_mat('data')
        corrMat.plot_corr_mat('diff')
        corrMat.plot_corr_mat('diffc')
        #corrMat.save(options.outdir, options.varrs)
        corrMat.save(options.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument('-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--data', action='store', type=str, required=True)
    requiredArgs.add_argument('-o', '--outdir', action='store', type=str, required=True)
    requiredArgs.add_argument('-E', '--EBEE', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    optionalArgs.add_argument('-w', '--no_reweight', action='store_true', default=False)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    optionalArgs.add_argument('-f', '--final_reg', action='store_true', default=False)
    options = parser.parse_args()
    main(options)
