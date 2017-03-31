# -*- coding: utf-8 -*-
import numpy as np
from sktensor import dtensor as dt
import copy
import itertools as it
import mkkernel as mkk
import MTGP
from hyperopt import fmin, tpe, hp, rand


def kouho_searcher_bo(nd2, mode, kern, kerngeno, genomat, kernenv, envmat, r2, parameter_candidates, nsample=100):
    def cross_validation_mtgp(args):
        dim = nd2.shape
        length = MTGP.prod(dim)
        rx = 1  ##Distincti from  outer
        r3 = copy.deepcopy(r2)
        if mode == 'env':

            nk = dim[0]
            print("Env")

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

        elif mode == 'geno':

            nk = dim[1]
            print("Geno")

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

        elif mode == 'outer':

            nk = dim[0]
            print("Outer")
            r3 = 2 * r3
            rx = r3  ##Only outer
            # rx=1 ##Distincti from  outer

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

            nk2 = dim[1]

            residual = int(nk2 % r3)
            sho = int(nk2 / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk2)
            stock2 = []
            for i in range(int(r3)):
                stock2.append(x[group[i]])

        elif mode == 'inner':
            nk = MTGP.prod(dim)
            print("Inner")
            # rx=1 ##Distincti from  outer

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])


        else:
            print('None Sence!!')

        print(args)
        print(args.values())

        argspre = sorted(args.items())
        argsuse = []
        for i in range(len(args.items())):
            argsuse.append(argspre[i][1])
        envpara, gpara, mparae, mparag, parse, parsg, parst, sig2 = argsuse

        print("envpara=" + str(envpara)[:3])
        print("gpara=" + str(gpara)[:3])

        rx2 = r3 * rx
        dim2 = list(copy.deepcopy(dim))
        dim2.append(rx2)
        scaled = np.tile([np.nan], dim2)
        est = np.tile([np.nan], (dim2))
        origin = np.tile([np.nan], (dim2))
        comp = np.tile([np.nan], (dim2))
        count = -1
        for p in range(r3):
            for q in range(rx):

                count += 1
                print(count)
                nd3 = copy.deepcopy(nd2)
                ##条件付け
                if mode == 'env':
                    u = stock[p]
                    nd3[u, :, :] = np.nan

                elif mode == 'geno':
                    u = stock[p]
                    nd3[:, u, :] = np.nan

                elif mode == 'outer':
                    u = stock[p]
                    nd3[u, :, :] = np.nan
                    u2 = stock2[q]
                    nd3[:, u2, :] = np.nan

                elif mode == 'inner':
                    u = stock[p]
                    vectenx = nd3.reshape(length, order='F')
                    vectenx[u] = np.nan
                    nd3 = vectenx.reshape(dim, order='F')

                vecten2 = nd3.reshape(length, order='F')
                full_site = np.logical_not(np.isnan(vecten2))
                pre = vecten2[full_site]
                vecten2[full_site] = pre
                miss_site = np.isnan(vecten2)
                nduse = vecten2.reshape(dim, order='F')

                sc = MTGP.scaling(nduse)
                nduse_sc = dt(sc['dtensor'])
                nduse_meansd = sc['mean_sd']
                nduse_impute = dt(mkk.ten_imputes(nduse_sc.__array__()))
                kerns = list()
                kerndic = mkk.selfkern(nd3, parsg, parse, parst, kern)
                kerndic = mkk.selfkern(nduse_impute, parsg, parse, parst, kern)  ##2015/1109に修正。なお、これで正常に動作することを確認。

                kerns.append(kerndic['env'])
                kerns.append(kerndic['geno'])
                kerns.append(kerndic['trait'])

                if genomat is not None:
                    gkern = kerngeno(genomat, gpara)
                    kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
                if envmat is not None:
                    ekern = kernenv(envmat, envpara)
                    kerns[0] = mparae * kerns[0] + (1 - mparae) * ekern
                solver = MTGP.GPI_normal(kerns, sig2, nduse_impute, miss_site)

                origin[:, :, :, count] = nd2
                nd_compare = MTGP.compare(nd2, nduse_meansd)
                comp[:, :, :, count] = nd_compare
                if mode != 'outer':
                    scaled[:, :, :, count] = solver.reshape(dim, order='F')
                    est[:, :, :, count] = MTGP.reconstruct(scaled[:, :, :, count], nduse_meansd)
                if mode == 'outer':
                    pre_solve = copy.deepcopy(solver.reshape(dim, order='F'))
                    medium = np.tile(np.nan, dim)
                    for w in u:
                        medium[w, u2, :] = pre_solve[w, u2, :]
                    scaled[:, :, :, count] = medium
                    est[:, :, :, count] = MTGP.reconstruct(scaled[:, :, :, count], nduse_meansd)

        mse = np.nanmean((comp - scaled) ** 2)

        print(mse)
        return (mse)

    def candidates_setting(mode, genomat, envmat):
        dparst = hp.loguniform("parst", np.log(0.01), np.log(10))
        dsig2 = hp.loguniform("sig2", np.log(0.01), np.log(100))

        global dparsg, dparse, dgpara, denvpara, dmparag, dmparae

        if (mode == 'inner'):
            if envmat is None:
                denvpara = hp.choice("envpara", [1])
                dmparae = hp.choice("mparae", [1])
            else:
                denvpara = hp.loguniform("envpara", np.log(0.01), np.log(10))
                dmparae = hp.uniform("mparae", 0, 1)

            if genomat is None:
                dgpara = hp.choice("gpara", [1])
                dmparag = hp.choice("mparag", [1])
            else:
                dgpara = hp.loguniform("gpara", np.log(0.01), np.log(10))
                dmparag = hp.uniform("mparag", 0, 1)

            dparse = hp.uniform("parse", 0, 1)
            dparsg = hp.uniform("parsg", 0, 1)

        if (mode == 'geno'):
            if envmat is None:
                denvpara = hp.choice("envpara", [1])
                dmparae = hp.choice("mparae", [1])
            else:
                denvpara = hp.loguniform("envpara", np.log(0.01), np.log(10))
                dmparae = hp.uniform("mparae", 0, 1)

            dgpara = hp.loguniform("gpara", np.log(0.01), np.log(10))
            dmparag = hp.choice("mparag", [0])

            dparse = hp.uniform("parse", 0, 1)
            dparsg = hp.choice("parsg", [1])

        if (mode == 'env'):

            denvpara = hp.loguniform("envpara", np.log(0.01), np.log(10))
            dmparae = hp.choice("mparae", [0])

            if genomat is None:
                dgpara = hp.choice("gpara", [1])
                dmparag = hp.choice("mparag", [1])
            else:
                dgpara = hp.loguniform("gpara", np.log(0.01), np.log(10))
                dmparag = hp.uniform("mparag", 0, 1)

            # dmparag=hp.uniform("mparag", 0,1)
            dparse = hp.choice("parse", [1])
            dparsg = hp.uniform("parsg", 0, 1)

        if (mode == 'outer'):
            denvpara = hp.loguniform("envpara", np.log(0.01), np.log(10))
            dmparae = hp.choice("mparae", [0])

            dgpara = hp.loguniform("gpara", np.log(0.01), np.log(10))
            dmparag = hp.choice("mparag", [0])

            # dmparag=hp.uniform("mparag", 0,1)
            dparse = hp.choice("parse", [1])
            dparsg = hp.choice("parsg", [1])

        parameter_kouho = {
            'parsg': dparsg,

            'parse': dparse,
            'parst': dparst,
            'gpara': dgpara,
            'envpara': denvpara,
            'sig2': dsig2,
            'mparag': dmparag,
            'mparae': dmparae
        }

        return (parameter_kouho)

    if parameter_candidates is None:
        parameter_candidates = candidates_setting(mode, genomat, envmat)

    best = fmin(cross_validation_mtgp, parameter_candidates, algo=tpe.suggest, max_evals=nsample)
    print("best is below")
    print(best)
    return (best)


def convert_params(best, mode, envmat, genomat):
    parst = best['parst']
    sig2 = best["sig2"]

    global parsg, parse, gpara, envpara, mparag, mparae

    # if (mode=='unif')|(mode=='fiber'):
    if (mode == 'inner'):
        if envmat is None:
            envpara = [1][best["envpara"]]
            mparae = [1][best["mparae"]]
        else:
            envpara = best["envpara"]
            mparae = best["mparae"]

        if genomat is None:
            gpara = [1][best["gpara"]]
            mparag = [1][best["mparag"]]
        else:
            gpara = best["gpara"]
            mparag = best["mparag"]

        parse = best["parse"]
        parsg = best["parsg"]

    if (mode == 'geno'):
        if envmat is None:
            envpara = [1][best["envpara"]]
            mparae = [1][best["mparae"]]
        else:
            envpara = best["envpara"]
            mparae = best["mparae"]

        gpara = best["gpara"]
        mparag = best["mparag"]

        parse = best["parse"]
        parsg = [0][best["parsg"]]

    if (mode == 'env'):

        envpara = best["envpara"]
        mparae = [0][best["mparae"]]

        if genomat is None:
            gpara = [1][best["gpara"]]
            mparag = [1][best["mparag"]]
        else:
            gpara = best["gpara"]
            mparag = best["mparag"]

        # dmparag=hp.uniform("mparag", 0,1)
        parse = [0][best["parse"]]
        parsg = best["parsg"]

    if (mode == 'outer'):
        envpara = best["envpara"]
        mparae = [0][best["mparae"]]

        gpara = best["gpara"]
        mparag = [0][best["mparag"]]

        # dmparag=hp.uniform("mparag", 0,1)
        parse = [0][best["parse"]]
        parsg = [0][best["parsg"]]

    parameter_kouho = {
        'parsg': parsg,  # 'parsg': hp.choice("parsg", [1]),
        'parse': parse,
        'parst': parst,
        'gpara': gpara,
        'envpara': envpara,
        'sig2': sig2,
        'mparag': mparag,
        'mparae': mparae
    }

    return (parameter_kouho)

def MTGP_impute_TPE(nd, kern, kerngeno, genomat, kernenv, envmat, mode, r2,
                    parameter_candidates, nsample=100):
    """
    Prediction function by using tree-structured parzen estimator (Bergstra et al., 2011)
    :param nd: N_E*N_G*N_T three-way nd-array phenotypic values data of N_E environments, N_G genotypes and N_T traits.
    :param kern: Kernel function of self-similarity matrixes
    :param kerngeno: Kernel function for SNPs marker matrix. If you do not have marker matrix, it should be None.
    :param genomat: SNPs marker matrix: N_G * Markers nd-array matrix
    :param kernenv: #Kernel function for additive information of environments.
    :param envmat:  #Additive information of genotype. If you do not have them, it should be None.
    :param mode: str object. There are three choices depends on the pattern of missing; "inner" phenotypic values are missing randomly; "geno" all phenotypic values of some genotypes are missing;, "env" all phenotypic values at some environments are missing
    :param r2: Replication number of CV. If you have plenty of time, it should be 10 or 20.
    :param  parameter_candidates: Candidate distribution of parameters. It can be set arbitrary by following the manner of hyperopt library. If it is none, the default setting is used. Python dictionary object which has indexes; ,'envpara', 'gpara', 'mparae', 'mparag', 'parse', 'parsg', 'parst', 'sig2''. Each index have distribution of parameters.

    :return Python dictionary object which has three indexes; 'result', 'est' and 'Parameters'. 'result' contains competed three-way array data. 'est' contains estimated missing data. 'parameters' contains parameters used for imputation.
    """

    length = MTGP.prod(nd.shape)
    dim = nd.shape

    result = copy.deepcopy(nd)
    nd2 = copy.deepcopy(nd)

    best = kouho_searcher_bo(nd2, mode, kern, kerngeno, genomat, kernenv, envmat, r2, parameter_candidates,
                             nsample=nsample)
    global a1
    if parameter_candidates is None:
        a1 = convert_params(best, mode, envmat, genomat)
    parse, parst, parsg, sig2, envpara, gpara, mparae, mparag = [ai[1] for ai in a1.items()]

    vecten2 = nd2.reshape(length, order='F')

    miss_site = np.isnan(vecten2)

    sc = MTGP.scaling(nd)
    nduse_sc = dt(sc['dtensor'])
    nduse_meansd = sc['mean_sd']
    nduse_impute = mkk.ten_imputes(nduse_sc.__array__())

    kerndic = mkk.selfkern(nduse_impute, parse, parst, parsg, kern)
    kerns = [kerndic['env'], kerndic['geno'], kerndic['trait']]

    if genomat is not None:  ##If you do not have marker genotype data.
        gkern = kerngeno(genomat, gpara)
        if mparag > 0:
            kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
        else:
            kerns[1] = gkern

    if envmat is not None:  ##If you do not have environmental covariates data.
        ekern = kerngeno(envmat, envpara)
        if mparag > 0:
            kerns[0] = mparae * (kerns[0]) + (1 - mparae) * ekern
        else:
            kerns[0] = ekern

    solver = MTGP.GPI_normal(kerns, sig2, nduse_impute, miss_site)

    scaled = solver.reshape(dim, order='F')
    est = MTGP.reconstruct(scaled, nduse_meansd)
    result[np.isnan(nd)] = est[np.isnan(nd)]

    rdic = {
        'result': result,  # Competed three-way array data
        'est': est,  # Estimated missing data
        'Parameters': a1  # Parameters used for imputation

    }

    return (rdic)

def MTGP_impute_TPE_precision_check(nd, kern, kerngeno, genomat, kernenv, envmat, mode, r2,
                    parameters):
    """
    Function to conduct cross validation for checking the precision accuracy
    :param nd: N_E*N_G*N_T three-way nd-array phenotypic values data of N_E environments, N_G genotypes and N_T traits.
    :param kern: Kernel function of self-similarity matrixes
    :param kerngeno: Kernel function for SNPs marker matrix. If you do not have marker matrix, it should be None.
    :param genomat: SNPs marker matrix: N_G * Markers nd-array matrix
    :param kernenv: #Kernel function for additive information of environments.
    :param envmat:  #Additive information of genotype. If you do not have them, it should be None.
    :param mode: str object. There are three choices depends on the pattern of missing; "inner" phenotypic values are missing randomly; "geno" all phenotypic values of some genotypes are missing;, "env" all phenotypic values at some environments are missing
    :param r2: Replication number of CV. If you have plenty of time, it should be 10 or 20.
    :param  parameters: Parameters estimated by MTGP_impute_TPE. They are Python dictionary object which has indexes; ,'envpara', 'gpara', 'mparae', 'mparag', 'parse', 'parsg', 'parst', 'sig2''. Each index have value of parameters.

    :return Python dictionary object which has two indexes; 'CV_result' and 'CV_presicion'. 'result' contains competed three-way array data. 'CV_presicion' contains estimated missing data. 'parameters' contains parameters used for imputation.
    """
    dim = nd.shape

    def cross_validation_mtgp(args):
        """
        Function to conduct cross validation for checking the precision accuracy
        :param args: Dictionary object of parameters
        :return:
        """
        nd2 = copy.deepcopy(nd)
        dim = nd2.shape
        length = MTGP.prod(dim)
        rx = 1  ##Distincti from  outer
        r3 = copy.deepcopy(r2)
        if mode == 'env':

            nk = dim[0]
            print("Env")

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

        elif mode == 'geno':

            nk = dim[1]
            print("Geno")

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

        elif mode == 'outer':

            nk = dim[0]
            print("Outer")
            r3 = 2 * r3
            rx = r3  ##Only outer
            # rx=1 ##Distincti from  outer

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])

            nk2 = dim[1]

            residual = int(nk2 % r3)
            sho = int(nk2 / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk2)
            stock2 = []
            for i in range(int(r3)):
                stock2.append(x[group[i]])

        elif mode == 'inner':
            nk = MTGP.prod(dim)
            print("Inner")
            # rx=1 ##Distincti from  outer

            residual = int(nk % r3)
            sho = int(nk / r3)
            group = []
            for i in range(int(r3)):
                group.append(list(range(sho * i, sho * (i + 1))))
            if residual != 0:
                for i in range(residual):
                    group[i].append(sho * r3 + i)
            np.random.seed(50)
            x = np.random.permutation(nk)
            stock = []
            for i in range(int(r3)):
                stock.append(x[group[i]])


        else:
            print('None Sence!!')

        #Setting parameters
        argspre = sorted(args.items())
        argsuse = [argi[1] for argi in argspre]
        envpara, gpara, mparae, mparag, parse, parsg, parst, sig2 = argsuse

        print("envpara=" + str(envpara)[:3])
        print("gpara=" + str(gpara)[:3])

        rx2 = r3 * rx
        dim2 = list(copy.deepcopy(dim))
        dim2.append(rx2)
        scaled = np.tile([np.nan], dim2)
        est = np.tile([np.nan], (dim2))
        origin = np.tile([np.nan], (dim2))
        comp = np.tile([np.nan], (dim2))
        count = -1
        for p in range(r3):
            for q in range(rx):

                count += 1
                print(count)
                nd3 = copy.deepcopy(nd2)

                #Making missing data
                ##条件付け
                if mode == 'env':
                    u = stock[p]
                    nd3[u, :, :] = np.nan

                elif mode == 'geno':
                    u = stock[p]
                    nd3[:, u, :] = np.nan

                elif mode == 'outer':
                    u = stock[p]
                    nd3[u, :, :] = np.nan
                    u2 = stock2[q]
                    nd3[:, u2, :] = np.nan

                elif mode == 'inner':
                    u = stock[p]
                    vectenx = nd3.reshape(length, order='F')
                    vectenx[u] = np.nan
                    nd3 = vectenx.reshape(dim, order='F')

                vecten2 = nd3.reshape(length, order='F')
                full_site = np.logical_not(np.isnan(vecten2))
                pre = vecten2[full_site]
                vecten2[full_site] = pre
                miss_site = np.isnan(vecten2)
                nduse = vecten2.reshape(dim, order='F')

                #Calculating self measuring similarity
                sc = MTGP.scaling(nduse)
                nduse_sc = dt(sc['dtensor'])
                nduse_meansd = sc['mean_sd']
                nduse_impute = dt(mkk.ten_imputes(nduse_sc.__array__()))
                kerns = list()
                kerndic = mkk.selfkern(nduse_impute, parsg, parse, parst, kern)

                kerns.append(kerndic['env'])
                kerns.append(kerndic['geno'])
                kerns.append(kerndic['trait'])

                #Calculating kernels from additive information and combine them with self measuring similarity
                if genomat is not None:
                    gkern = kerngeno(genomat, gpara)
                    kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
                if envmat is not None:
                    ekern = kernenv(envmat, envpara)
                    kerns[0] = mparae * kerns[0] + (1 - mparae) * ekern

                #Estimating missing values
                solver = MTGP.GPI_normal(kerns, sig2, nduse_impute, miss_site)
                origin[:, :, :, count] = nd2
                nd_compare = MTGP.compare(nd2, nduse_meansd)
                comp[:, :, :, count] = nd_compare
                if mode != 'outer':
                    scaled[:, :, :, count] = solver.reshape(dim, order='F')
                    est[:, :, :, count] = MTGP.reconstruct(scaled[:, :, :, count], nduse_meansd)
                if mode == 'outer':
                    pre_solve = copy.deepcopy(solver.reshape(dim, order='F'))
                    medium = np.tile(np.nan, dim)
                    for w in u:
                        medium[w, u2, :] = pre_solve[w, u2, :]
                    scaled[:, :, :, count] = medium
                    est[:, :, :, count] = MTGP.reconstruct(scaled[:, :, :, count], nduse_meansd)

        mse = np.nanmean((comp - scaled) ** 2)

        print('Current mse of scaled values is'+str(mse)[:4])
        return (est)

    def check_precision(nd,est_CV):
        """
        Function to calculate the precison of imputation
        :param nd: Original missing nd array data
        :param est_CV: Estimated values in cross validation by "cross_validation_mtgp" above
        :return:N_E×N_T matrix of precision, which corresponds to the precision of each trait at each environment. R^2 between observed values and predicted values is used as the precision.
        """
        def R_sq(nd1,nd2):
            pos1=~np.isnan(nd1)
            pos2=~np.isnan(nd2)
            pos=pos1&pos2
            r_sq=np.corrcoef(nd1[pos],nd2[pos])[1,0]**2
            return(r_sq)

        nds=np.tile(np.nan,list(dim)+[r2])
        for i in range(r2):
            nds[:,:,:,i]=nd

        Precison_matrix=np.tile(np.nan,[dim[0],dim[2]])
        index=it.product(range(dim[0]),range(dim[2]))
        for i,j in index:
            r_sq_ij=R_sq(nds[i,:,j,:].reshape(dim[1]*r2),est_CV[i,:,j,:].reshape(dim[1]*r2))
            Precison_matrix[i,j]=r_sq_ij

        return(Precison_matrix)

    if parameters is None:
        return("Parameters are empty!")

    est_CV=cross_validation_mtgp(parameters)
    precision=check_precision(nd,est_CV)

    rdic = {
        'CV_result': est_CV,  # Estimated values in cross validation
        'CV_presicion': precision,  # Estimated precision
    }

    return (rdic)
