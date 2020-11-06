import gzip
import os
#import ROOT as rt
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle as pkl

from joblib import delayed, Parallel, parallel_backend, register_parallel_backend
from dask.distributed import Client, LocalCluster, progress, wait, get_client
from dask_jobqueue import SLURMCluster

from sklearn.ensemble import GradientBoostingRegressor
from .tmva.IdMVAComputer import IdMvaComputer, helpComputeIdMva
from .tmva.eleIdMVAComputer import eleIdMvaComputer, helpComputeEleIdMva
from .Corrector import Corrector, applyCorrection
from .quantileRegression_chain import quantileRegression_chain, trainClf
from .Shifter import Shifter, applyShift
from .Shifter2D import Shifter2D, apply2DShift

import logging
logger = logging.getLogger(__name__)


class quantileRegression_chain_disc(quantileRegression_chain):

    def trainp0tclf(self,var,key,weightsDir ='weights_qRC',n_jobs=1):

        if key == 'mc':
            df = self.MC
        elif key == 'data':
            df = self.data
        else:
            raise KeyError('Please use data or mc')

        features = self.kinrho

        weightsDir = weightsDir if weightsDir.startswith('/') else '{}/{}'.format(self.workDir, weightsDir)

        df['p0t_{}'.format(var)] = np.apply_along_axis(lambda x: 0 if x==0 else 1,0,df[var].values.reshape(1,-1))
        X = df.loc[:,features].values
        Y = df['p0t_{}'.format(var)].values
        clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,maxDepth=10,subsample=0.5,gamma=0, n_jobs=n_jobs)
        clf.fit(X, Y)

        X_names = features
        Y_name = var
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}_clf_p0t_{}_{}.pkl'.format(weightsDir,key,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

    def train3Catclf(self,varrs,key,weightsDir='weights_qRC',n_jobs=1):

        if key == 'mc':
            df = self.MC
        elif key == 'data':
            df = self.data
        else:
            raise KeyError('Please use data or mc')

        features = self.kinrho

        weightsDir = weightsDir if weightsDir.startswith('/') else '{}/{}'.format(self.workDir, weightsDir)

        df['ChIsoCat'] = self.get_class_3Cat(df[varrs[0]].values,df[varrs[1]].values)
        X = df.loc[:,features].values
        Y = df['ChIsoCat'].values
        clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, maxDepth=10,gamma=0, n_jobs=n_jobs)
        clf.fit(X, Y)

        X_names = features
        Y_names = [varrs[0],varrs[1]]
        dic = {'clf': clf, 'X': X_names, 'Y': Y_names}
        pkl.dump(dic,gzip.open('{}/{}_clf_3Cat_{}_{}_{}.pkl'.format(weightsDir,key,self.EBEE,varrs[0],varrs[1]),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

    def get_class_3Cat(self,x,y):
        return [0 if x[i]==0 and y[i]==0 else (1 if x[i]==0 and y[i]>0 else 2) for i in range(len(x))]

    def load3Catclf(self,varrs,weightsDir='weights_qRC'):

        self.TCatclf_mc = self.load_clf_safe(varrs,weightsDir,'mc_clf_3Cat_{}_{}_{}.pkl'.format(self.EBEE,varrs[0],varrs[1]),self.kinrho)
        self.TCatclf_d = self.load_clf_safe(varrs,weightsDir,'data_clf_3Cat_{}_{}_{}.pkl'.format(self.EBEE,varrs[0],varrs[1]),self.kinrho)

    def trainTailRegressors(self,var,weightsDir='weights_qRC'):

        features = self.kinrho+['{}'.format(x) for x in self.vars if not x == var]
        X = self.MC.query('{}!=0'.format(var)).loc[:,features].values
        Y = self.MC.query('{}!=0'.format(var))[var].values

        futures = [self.client.submit(trainClf,
            quantile ,
            5,
            500,
            X,
            Y,
            save=True,
            outDir=weightsDir if weightsDir.startswith('/') else '{}/{}'.format(
                self.workDir,weightsDir),
            name='mc_weights_tail_{}_{}_{}'.format(
                self.EBEE,var,str(quantile).replace('.','p')),
            X_names=features,
            Y_name=var) for quantile in self.quantiles
            ]

        return futures

    def loadTailRegressors(self,varrs,weightsDir):

        if not isinstance(varrs,list):
            varrs = list((varrs,))
        self.tail_clfs_mc = {}
        for var in varrs:
            self.tail_clfs_mc[var] = [self.load_clf_safe(var, weightsDir,'mc_weights_tail_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p')),self.kinrho+[x for x in self.vars if not x == var]) for q in self.quantiles]

    def loadp0tclf(self,var,weightsDir):

        self.p0tclf_mc = self.load_clf_safe(var, weightsDir,'mc_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))
        self.p0tclf_d = self.load_clf_safe(var, weightsDir,'data_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))

    def shiftY(self,var,finalReg=False,n_jobs=1):

        features = self.kinrho
        X = self.MC.loc[:,features]
        Y = self.MC[var]

        Y = Y.values.reshape(-1,1)
        Z = np.hstack([X,Y])

        logger.info('Shifting {} with input features {}'.format(var,features))

        n_workers = len(get_client().scheduler_info()['workers'])

        if finalReg:
            future_splits = [self.client.submit(applyShift,
                    self.p0tclf_mc,
                    self.p0tclf_d,
                    [self.finalTailRegs[var]],
                    ch[:,:-1],
                    ch[:,-1]) for ch in np.array_split(Z, n_workers)]
            progress(future_splits)
            Y_shift = np.concatenate(self.client.gather(future_splits))
            self.MC['{}_shift_final'.format(var)] = Y_shift
            del future_splits
        else:
            future_splits = [self.client.submit(applyShift,
                    self.p0tclf_mc,
                    self.p0tclf_d,
                    self.clfs_mc,
                    ch[:,:-1],
                    ch[:,-1]) for ch in np.array_split(Z, n_workers)]
            progress(future_splits)
            Y_shift = np.concatenate(self.client.gather(future_splits))
            self.MC['{}_shift'.format(var)] = Y_shift
            del future_splits


    def shiftY2D(self,varrs,finalReg=False,n_jobs=1):

        features = self.kinrho
        X = self.MC.loc[:,features]
        Y = self.MC.loc[:,varrs]

        if X.isnull().values.any():
            raise KeyError('Correct all of {} first!'.format(varrs))

        Y = Y.values.reshape(-1,2)
        Z = np.hstack([X,Y])

        n_workers = len(get_client().scheduler_info()['workers'])

        if finalReg:
            future_splits = [self.client.submit(apply2DShift,
                    self.TCatclf_mc,
                    self.TCatclf_d,
                    [self.finalTailRegs[varrs[0]]],
                    [self.finalTailRegs[varrs[1]]],
                    ch[:,:-2],
                    ch[:,-2:]) for ch in np.array_split(Z, n_workers)]
            progress(future_splits)
            Y_shift = np.concatenate(self.client.gather(future_splits))
            self.MC['{}_shift_final'.format(varrs[0])] = Y_shift[:,0]
            self.MC['{}_shift_final'.format(varrs[1])] = Y_shift[:,1]
        else:
            future_splits = [self.client.submit(apply2DShift,
                    self.TCatclf_mc,
                    self.TCatclf_d,
                    self.tail_clfs_mc[varrs[0]],
                    self.tail_clfs_mc[varrs[1]],
                    ch[:,:-2],
                    ch[:,-2:]) for ch in np.array_split(Z, n_workers)]
            progress(future_splits)
            Y_shift = np.concatenate(self.client.gather(future_splits))
            self.MC['{}_shift'.format(varrs[0])] = Y_shift[:,0]
            self.MC['{}_shift'.format(varrs[1])] = Y_shift[:,1]


    def correctY(self, var):

        if len(self.vars)==1:
            self.shiftY(var)
        elif len(self.vars)>1 and '{}_shift'.format(var) not in self.MC.columns:
            self.shiftY2D(self.vars)
        super(quantileRegression_chain_disc, self).correctY('{}_shift'.format(var), diz=True)

    def applyFinalRegression(self,var,n_jobs=1):

        if len(self.vars)==1:
            self.shiftY(var,finalReg=True,n_jobs=n_jobs)
        elif len(self.vars)>1 and '{}_shift_final'.format(var) not in self.MC.columns:
            self.shiftY2D(self.vars,finalReg=True,n_jobs=n_jobs)
        super(quantileRegression_chain_disc, self).applyFinalRegression('{}_shift_final'.format(var), diz=True)

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):

        logger.info('Training quantile regressors on data')
        return self._trainQuantiles('data_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        return self._trainQuantiles('mc_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def trainAllData(self, weightsDir):
        trained_regressors = super().trainAllData(weightsDir)
        if 'probeChIso03' in self.vars:
            if not os.path.exists('{}/{}/data_clf_3cat_{}_{}_{}.pkl'.format(
            self.workDir, weightsDir, self.EBEE, 'probeChIso03', 'probeChIso03worst')):
                logger.info('Applying train3Catclf')
                trained_regressors.append(self.train3Catclf(
                        ['probeChIso03', 'probeChIso03worst'],
                        'data',
                        weightsDir=weightsDir
                        ))
        if 'probePhoIso' in self.vars:
            if not os.path.exists('{}/{}/data_clf_p0t_{}_{}.pkl'.format(
                self.workDir, weightsDir, self.EBEE, 'probePhoIso')):
                logger.info('Applying trainp0tclf')
                trained_regressors.append(self.trainp0tclf(
                        'probePhoIso',
                        'data',
                        weightsDir=weightsDir))
        return trained_regressors

    def trainAllMC(self,weightsDir):

        # Train tail regressors
        try:
            self.loadTailRegressors(self.vars,weightsDir)
        except IOError:
            futures = [future for var in self.vars for future in self.trainTailRegressors(
                var, weightsDir)]

            logger.info('Waiting for MC trained tail regressors for vars {}'.format(
                self.vars))
            progress(futures)
            wait(futures)
            del futures

            self.loadTailRegressors(self.vars,weightsDir)

        for var in self.vars:
            try:
                self.loadClfs(var,weightsDir)
            except IOError:
                futures = self.trainOnMC(var,weightsDir=weightsDir)

                logger.info('Waiting for MC trained regressors for {}'.format(var))
                progress(futures)
                wait(futures)
                del futures

                self.loadClfs(var,weightsDir)

            try:
                if len(self.vars)>1:
                    self.load3Catclf(self.vars,weightsDir)
                else:
                    self.loadp0tclf(var,weightsDir)
            except IOError:
                if len(self.vars)>1:
                    self.train3Catclf(self.vars,key='mc',weightsDir=weightsDir)
                    self.load3Catclf(self.vars,weightsDir)
                else:
                    self.trainp0tclf(var,key='mc',weightsDir=weightsDir)
                    self.loadp0tclf(var,weightsDir)

                logger.info('Correcting variable {}'.format(var))
                self.correctY(var)

    def trainFinalRegression(self,var,weightsDir,n_jobs=1):
        super(quantileRegression_chain_disc,self).trainFinalRegression(var,weightsDir,diz=True,n_jobs=n_jobs)

    def trainFinalTailRegressor(self,var,weightsDir,weightsDirIn,n_jobs=1):

        if len(self.vars) == 1:
            self.loadClfs(var,weightsDirIn)
            self.MC.loc[self.MC[var] != 0,'cdf_{}'.format(var)] = self._getCondCDF(self.MC.loc[self.MC[var] != 0,:],self.clfs_mc,self.kinrho,var)
        elif len(self.vars) > 1:
            self.loadTailRegressors(self.vars,weightsDirIn)
            self.MC.loc[self.MC[var] != 0,'cdf_{}'.format(var)] = self._getCondCDF(self.MC.loc[self.MC[var] != 0,:],self.tail_clfs_mc[var],self.kinrho+[x for x in self.vars if not x == var],var)

        features = self.kinrho + [x for x in self.vars if not x == var] + ['cdf_{}'.format(var)]
        X = self.MC.loc[self.MC[var] != 0., features].values
        Y = self.MC.loc[self.MC[var] != 0., var].values

        logger.info('Training final tail regressor with features {} for {}'.format(features,var))
        clf = xgb.XGBRegressor(n_estimators=1000, maxDepth=10, gamma=0, n_jobs=n_jobs, base_score=0.)
        clf.fit(X,Y)

        name = 'weights_finalTailRegressor_{}_{}'.format(self.EBEE,var)
        dic = {'clf': clf, 'X': features, 'Y': var}
        pkl.dump(dic,gzip.open('{}/{}/{}.pkl'.format(self.workDir,weightsDir,name),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

    def loadFinalTailRegressor(self,varrs,weightsDir):

        if not isinstance(varrs,list):
            varrs = list((varrs,))
        self.finalTailRegs = {}
        for var in varrs:
            self.finalTailRegs[var] = self.load_clf_safe(var,weightsDir,'weights_finalTailRegressor_{}_{}.pkl'.format(self.EBEE,var),self.kinrho+[x for x in varrs if not x == var]+['cdf_{}'.format(var)],var)
