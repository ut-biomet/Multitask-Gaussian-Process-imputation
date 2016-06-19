# 2016/06/18 Trial of MTGP in MTGP_imputation directory #

import pandas as pd
import mktensor as mt
import mkkernel as mkk
import MTGP as MG
import os
import numpy as np

Dir_name = "tests"
Dir_trait="tests/traits"

filenames = os.listdir(Dir_trait)
ngeno = Dir_name + "/genotype_marker_sample.csv"
genos = pd.read_csv(ngeno, index_col=0).ix[0:30, 0:200].values

# Actual treatment#
ntrait = [Dir_trait + "/" + filenames[i] for i in [0] + range(2, len(filenames))]
tendata = mt.make_tensor_data(ntrait)

##
miss_rate = 0.5
length = tendata.shape[0] * tendata.shape[1] * tendata.shape[2]
pos = np.random.permutation(range(length))[0:int(length * miss_rate)]
tenvec_miss = tendata.reshape(length, order="F")
tenvec_miss[pos] = np.nan
tendata_miss = np.reshape(tenvec_miss, tendata.shape, order="F")

nd, kern, kerngeno, genomat, mode, r2 = tendata_miss, mkk.Gauss_kernel, mkk.median_RBF_kernel, genos, "Inner", 3
parameter_kouho = {
    'gpara_kouho': [2],
    'mparag_kouho': [i * 0.1 for i in range(1, 9)],
    'pars_kouho': [0.01, 0.1, 1, 10],
    'sig2_kouho': [0.01, 0.1, 1, 10]
}

MTGP1 = MG.MTGP_impute(nd, kern, kerngeno, genomat, mode, r2, parameter_kouho)

from matplotlib import pyplot as plt
plt.plot(tendata[np.isnan(tendata_miss)],MTGP1['est'][np.isnan(tendata_miss)],'o')
