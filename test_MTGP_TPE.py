# -*- coding: utf-8 -*-
import pandas as pd
from MTGP_imputation import mktensor as mt
from MTGP_imputation import mkkernel as mkk
from MTGP_imputation import MTGP_TPE as MGT
import os
import numpy as np
import sys

Dir_name = "tests" #Directory of test data
Dir_trait="tests/traits" #Directory of phenotypic value data

filenames = os.listdir(Dir_trait)
ngeno = Dir_name + "/genotype_marker_sample.csv" #Directory of genotype marker data
genos = pd.read_csv(ngeno, index_col=0).ix[0:30, 0:200].values

# Actual data#
ntrait = [Dir_trait + "/" + filenames[i] for i in [0] + range(2, len(filenames))]
tensor_data = mt.make_tensor_data(ntrait)

#Making missing data#
miss_rate = 0.1
length = tensor_data.shape[0] * tensor_data.shape[1] * tensor_data.shape[2]
pos = np.random.permutation(range(length))[0:int(length * miss_rate)]
tenvec_miss = tensor_data.reshape(length, order="F")
tenvec_miss[pos] = np.nan
tensor_data_miss = np.reshape(tenvec_miss, tensor_data.shape, order="F")

#Setting parameters#
nd =tensor_data_miss
kern=mkk.Gauss_kernel #Kernel function for self-measuring similarity
kerngeno=mkk.Gauss_kernel #Kernel function for additive information of genotype
genomat=genos #Additive information of genotype. If you do not have them, it should be None.
mode="inner" #Distribution of missing data. There are three choices depends on the pattern of missing; "inner" phenotypic values are missing randomly; "geno" all phenotypic values of some genotypes are missing;, "env" all phenotypic values at some environments are missing
r2 = 3 #The number of cross validation for parameter estimation. If you have plenty of time, it should be 10 or 20.
parameter_kouho = None #Candidate distribution of parameters. It can be set arbitrary by following the manner of hyperopt library. If it is none, the default setting is used.
kernenv=None #Kernel function for additive information of environments.
envmat= None #Additive information of genotype. If you do not have them, it should be None.
nsample=100 #Times of sampling TPE. If you have plenty of time, More than 49 are recommended.

MTGP1 = MGT.MTGP_impute_TPE(nd, kern, kerngeno, genomat,kernenv, envmat, mode, r2, parameter_kouho,nsample=nsample) # Parameter estimation and imputaion.

parameters=MTGP1['Parameters'] #Estimated parameters
MTGP_cv=MGT.MTGP_impute_TPE_precision_check(nd, kern, kerngeno, genomat, kernenv, envmat, mode, r2,
                    parameters) #Checking the imputation accuracy
print(MTGP_cv['CV_presicion'])

try:
    from matplotlib import pyplot as plt
    plt.plot(tensor_data[np.isnan(tensor_data_miss)],MTGP1['est'][np.isnan(tensor_data_miss)],'o')
except ImportError:
    sys.exit('could not load matplotlib library')
