# -*- coding: utf-8 -*-
import numpy as np
from sktensor import dtensor as dt
import copy
import itertools as it
import os
import mkkernel as mkk

def prod(seq):
    prod=1
    for element in seq:
        prod=prod*element
    return(prod)

def rev_kronecker_list(lis):
    len_list = len(lis)
    mat = lis[0]
    for i in range(len(lis) - 1):
        mat = np.kron(lis[i + 1], mat)
    return (mat)

def scaling(nduse):
    dim = list(nduse.shape)
    mean_sd = np.zeros((2, dim[2]))
    nduse_sc = np.tile(np.nan, dim)
    for i in range(dim[2]):
        sdi = np.nanstd(nduse[:, :, i])
        meani = np.nanmean(nduse[:, :, i])
        mean_sd[0, i] = meani
        mean_sd[1, i] = sdi
        nduse_sc[:, :, i] = (nduse[:, :, i] - meani) / sdi
    rdic = {}
    rdic['mean_sd'] = mean_sd
    rdic['dtensor'] = nduse_sc
    return (rdic)

def compare(nd, nduse_meansd):
    dim = nd.shape
    compare = np.tile(np.nan, dim)
    for i in range(dim[2]):
        compare[:, :, i] = (nd[:, :, i] - nduse_meansd[0, i]) / nduse_meansd[1, i]
    return (compare)


def GPI_normal_true2(kerns, sig2, nduse_impute,miss_site):
    """
    :type cov: object
    """
    dim = nduse_impute.shape

    vecten=nduse_impute.reshape(prod(dim),order='F')
    posi=miss_site
    posiuse=np.logical_not(posi)

    print('x')
    kf=rev_kronecker_list(kerns)+np.identity(prod(dim),dtype=np.float)*sig2
    kfuse=kf[posiuse,:][:,posiuse]
    cov=kf[posi,:][:,posiuse]
    solver1=np.linalg.solve(kfuse,vecten[posiuse])
    solver2=np.dot(cov,solver1)

    print(solver2.shape)
    solver=np.tile(np.nan,prod(dim))
    solver[posi]=solver2
    return (solver)

def reconstruct(ten, nduse_meansd):
    dim = ten.shape
    ten_origin = np.tile(np.nan, dim)
    for i in range(dim[2]):
        ten_origin[:, :, i] = ten[:,:,i] * nduse_meansd[1, i] + nduse_meansd[0, i]
    return(ten_origin)

def kouho_searcher(nd2, mode, kern, kerngeno, genomat,  r2, parameter_kouho):
    """

    :param nd2: three mode tenosr array data
    :param mode: str object which indicates the scenario of data missing.
    :param kern:
    :param kerngeno:
    :param genomat:
    :param r2:
    :param parameter_kouho:

    :return: test error of CV
    """

    global kouho

    kouhopre = sorted(parameter_kouho.items())
    bestuse = [kouhopre[i][1] for i in range(len(kouhopre))]

    gpara_kouho, mparag_kouho, pars_kouho, sig2_kouho = bestuse
    nd_l = copy.deepcopy(nd2)
    dim = nd_l.shape
    dim2 = list(copy.deepcopy(dim))
    dim2.append(r2)
    length = prod(dim)

    mse_top = np.array(1000000.)
    if mode == 'uniform':
        nk = prod(dim)
        print("uniform")

        residual = int(nk % r2)
        sho = int(nk / r2)
        group = []
        for i in range(int(r2)):
            group.append(list(range(sho * i, sho * (i + 1))))
        if residual != 0:
            for i in range(residual):
                group[i].append(sho * r2 + i)
        np.random.seed(50)
        x = np.random.permutation(nk)
        stock = []
        for i in range(int(r2)):
            stock.append(x[group[i]])
    elif mode == 'fiber':
        nk = dim[0]*dim[1]
        print("fiber")

        residual = int(nk % r2)
        sho = int(nk / r2)
        group = []
        for i in range(int(r2)):
            group.append(list(range(sho * i, sho * (i + 1))))
        if residual != 0:
            for i in range(residual):
                group[i].append(sho * r2 + i)
        np.random.seed(50)
        x = np.random.permutation(nk)
        stock = []
        for i in range(int(r2)):
            stock.append(x[group[i]])

    else:
        print('None Sence!!')

    for pars, gpara, mparag, sig2 in list(it.product(pars_kouho,gpara_kouho, mparag_kouho, sig2_kouho)):
        # pars = pars_kouho[npars]
        # gpara = gpara_kouho[ngpara]
        # envpara = envpara_kouho[nepara]
        # mparag = mparag_kouho[nmparag]
        # mparae = mparae_kouho[nmparae]
        # sig2=sig2_kouho[nsig2]

        scaled = np.tile([np.nan], (dim2))
        est = np.tile([np.nan], (dim2))
        origin = np.tile([np.nan], (dim2))
        comp = np.tile([np.nan], (dim2))

        for p in range(r2):
            print(p)
            ndp = copy.deepcopy(nd_l)

            if mode == 'uniform':
                u = stock[p]
                vectenx = ndp.reshape(length, order='F')
                vectenx[u] = np.nan
                nd3 = vectenx.reshape(dim, order='F')

            elif mode == 'fiber':
                u = stock[p]
                vectenx = ndp.reshape([dim[0]*dim[1],dim[2]], order='F')
                vectenx[u,:] = np.nan
                nd3 = vectenx.reshape(dim, order='F')

            vecten2 = nd3.reshape(length, order='F')
            miss_site = np.isnan(vecten2)
            nduse = vecten2.reshape(dim, order='F')

            sc = scaling(nduse)
            nduse_sc = dt(sc['dtensor'])
            nduse_meansd = sc['mean_sd']
            nduse_impute = dt(mkk.ten_imputes(nduse_sc.__array__()))
            kerns = list()
            for i in range(len(dim)):
                unfoldi = nduse_impute.unfold(i)
                kerns.append(kern(unfoldi.__array__(), pars))

            if genomat is not None:
                gkern = kerngeno(genomat, gpara)
                kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
            solver = GPI_normal_true2(kerns, sig2, nduse_impute, miss_site)

            origin[:, :, :, p] = nd2
            nd_compare = compare(nd2, nduse_meansd)
            comp[:, :, :, p] = nd_compare
            scaled[:, :, :, p] = solver.reshape(dim, order='F')
            est[:, :, :, p] = reconstruct(scaled[:, :, :, p], nduse_meansd)
        mse = np.nanmean((scaled - comp) ** 2) ** 0.5

        if mse < mse_top:
            kouho = [gpara, mparag, pars, sig2]
            mse_top = copy.deepcopy(mse)
    print(mse_top)

    return kouho

def MTGP_impute(nd, kern,  kerngeno, genomat,mode, r2,
                                        parameter_kouho):
    """

    :type nd: nd array object
    :param nd: N_E*N_G*N_T three-way nd-array data: N_E environments, N_G genotypes and N_T traits.
    :param kern: Kernel function for self-similarity matrix
    :param name: Name of result-folder :str
    :param kerngeno: Kernel function for SNPs marker matrix
    :param genomat: SNPs marker matrix: N_G * Markers nd-array matrix
    :param mode: How to make missing data for cross validation (CV). "uniform" or "fiber"
    :param r2: Replication number of CV :int
    :param parameter_kouho: python dictionary object which has indexes "    'gpara_kouho', 'mparag_kouho','pars_kouho','sig2_kouho'. Each index have list of values for cross validation.

    :return:
    """

    length = prod(nd.shape)
    dim = nd.shape
    kouho = []

    result=copy.deepcopy(nd)
    nd2 = copy.deepcopy(nd)

    best = kouho_searcher(nd2, mode, kern, kerngeno, genomat,  r2, parameter_kouho)
    gpara, mparag, pars, sig2 = best

    kouho.append(best)  ##parstを追加
    vecten2 = nd2.reshape(length, order='F')

    miss_site = np.isnan(vecten2)

    sc = scaling(nd)
    nduse_sc = dt(sc['dtensor'])
    nduse_meansd = sc['mean_sd']
    nduse_impute = mkk.ten_imputes(nduse_sc.__array__())

    kerndic = mkk.selfkern(nduse_impute, pars, pars, pars, kern)
    kerns = [kerndic['env'],kerndic['geno'],kerndic['trait']]

    if genomat is not None:  ##系統の付加情報
        gkern = kerngeno(genomat, gpara)
        if mparag > 0:
            kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
        else:
            kerns[1] = gkern

    solver = GPI_normal_true2(kerns, sig2, nduse_impute, miss_site)

    scaled = solver.reshape(dim, order='F')
    est = reconstruct(scaled, nduse_meansd)
    result[np.isnan(nd)]=est[np.isnan(nd)]

    rdic = {
        'result': result,
        'est': est,
        'kouhos': kouho
    }

    return (rdic)
