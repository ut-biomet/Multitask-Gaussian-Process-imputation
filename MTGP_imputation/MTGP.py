# -*- coding: utf-8 -*-
import numpy as np
from sktensor import dtensor as dt
import copy
import itertools as it
import os
import mkkernel as mkk

def prod(seq):
    """
    Function to calculate the product of numbers in list
    :type seq: list or ndarray
    :param seq: list object which contains numbers
    :rtype float
    :return The product of all the numbers in seq.
    """
    prod=1
    for element in seq:
        prod=prod*element
    return(prod)

def rev_kronecker_list(lis):
    """
    Function to calculate the kronecker product of matrixes in "lis" in reverse order.
    :type lis: list
    :param lis: List of 3 nd array matrixes. The size of matrix should be N_E*N_E, N_G*N_G and N_T*N_T.
    :rtype: ndarray
    :return: The kronecker product nd array matrix with the size of (N_T*N_G*N_E)*(N_T*N_G*N_E).
    """
    len_list = len(lis)
    mat = lis[0]
    for i in range(len_list - 1):
        mat = np.kron(lis[i + 1], mat)
    return (mat)

def scaling(nduse):
    """
    Function to standardize three-way array phenotypic values data as equation (1) does.
    Phenotypic values of each trait are standardized so that they hace a mean of 0 and SD of 1 over G genotype and R environments for each trait.
    Means, SDs of each trait and stasndardized three-way array data are returned.
    :type nduse: ndarray
    :param nduse: Three-way nd array object with the size of N_E environments, N_G genontypes and N_T traits.
    :rtype: dict
    :return: Python dictionary object which have two indexes; 'mean_sd' and 'dtensor'. 'mean_sd' contains means and SDs of each trai. 'dtensor' contsins standardized three-way array data.
    """
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
    """
    Function to calculate standardized data.
    :type type: ndarray
    :type nd: ndarray
    :param nd: Original missing data
    :param nduse_meansd:
    :rtype: ndarray
    :return: Standardized data
    """
    dim = nd.shape
    compare = np.tile(np.nan, dim)
    for i in range(dim[2]):
        compare[:, :, i] = (nd[:, :, i] - nduse_meansd[0, i]) / nduse_meansd[1, i]
    return (compare)


def GPI_normal(kerns, sig2, nduse_impute,miss_site):
    """
    Function to solve the equation (3) in the paper.
    :param kerns: List object which contains 3 nd array matrixes. The size of matrix should be N_E*N_E, N_G*N_G and N_T*N_T.
    :param sig2: Positive nd array value which corresponds to "sigma^2" (variance of error tertm) in the paper.
    :param nduse_impute: Missing three-way nd array to be imputed with the size of N_E environments, N_G genontypes and N_T traits.
    :param miss_site: nd array vector which indicates where vectorized phenotypic values are missing.
    :return: nd array vector of predicted valued for missing phenotypic values.
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
    """
    Function to transforming back standardized phenotypic values
    :type ten: ndarray
    :param ten: Imputed standardized phenotypic values
    :type nduse_meansd: ndarray
    :param nduse_meansd: Means and SDs of original missing data
    :rtype: ndarray
    :return: Transformed back phenotypic values.
    """
    dim = ten.shape
    ten_origin = np.tile(np.nan, dim)
    for i in range(dim[2]):
        ten_origin[:, :, i] = ten[:,:,i] * nduse_meansd[1, i] + nduse_meansd[0, i]
    return(ten_origin)

def candidates_searcher(nd2, mode, kern, kerngeno, genomat,  r2, parameter_candidates):
    """
    Function to determine parameters via cross validation (CV) for imputation.
    :param nd2: Standardized three-way nd array object with the size of N_E environments, N_G genontypes and N_T traits.
    :param mode: str object which indicates the scenario of data missing. It should be 'unif' or 'fiber'
    :param kern: Kernel function of self measuring similarity kernels.
    :param kerngeno: Kernel function of kernels of genotype marker matrix.
    :param genomat: nd array matrix of genotype marker data.
    :param r2: Integer which indicates the number of cross validation. We set it 'three' in the paper.
    :param parameter_candidates: python dictionary object which has indexes; 'gpara_candidates', 'mparag_candidates','pars_candidates','sig2_candidates'. Each index have list of values for cross validation.

    :return: Parameters which minimize CV error.
    """

    global candidates

    candidatespre = sorted(parameter_candidates.items())
    bestuse = [candidatespre[i][1] for i in range(len(candidatespre))]

    gpara_candidates, mparag_candidates, pars_candidates, sig2_candidates = bestuse
    nd_l = copy.deepcopy(nd2)
    dim = nd_l.shape
    dim2 = list(copy.deepcopy(dim))
    dim2.append(r2)
    length = prod(dim)

    mse_top = np.array(1000000.)
    cv_lacked=nd2
    cv_est=None

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
        print('None Sence!! "mode" should be "fiber", or "unif".')

    for pars, gpara, mparag, sig2 in list(it.product(pars_candidates,gpara_candidates, mparag_candidates, sig2_candidates)):

        scaled = np.tile([np.nan], (dim2))
        est = np.tile([np.nan], (dim2))
        origin = np.tile([np.nan], (dim2))
        comp = np.tile([np.nan], (dim2))
        lacked=np.tile([np.nan], (dim2))

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
            solver = GPI_normal(kerns, sig2, nduse_impute, miss_site)

            origin[:, :, :, p] = nd2
            nd_compare = compare(nd2, nduse_meansd)
            comp[:, :, :, p] = nd_compare
            scaled[:, :, :, p] = solver.reshape(dim, order='F')
            est[:, :, :, p] = reconstruct(scaled[:, :, :, p], nduse_meansd)
            lacked[:,:,:,p]=nduse
        mse = np.nanmean((scaled - comp) ** 2) ** 0.5

        if mse < mse_top:
            candidates = [gpara, mparag, pars, sig2]
            mse_top = copy.deepcopy(mse)
            cv_est=est
        cv_lacked=lacked

    print('The least error is below.')
    print(mse_top)

    return candidates+[cv_est,cv_lacked]

def MTGP_impute(nd, kern,  kerngeno, genomat,mode, r2,
                                        parameter_candidates):
    """

    :param nd: N_E*N_G*N_T three-way nd-array phenotypic values data of N_E environments, N_G genotypes and N_T traits.
    :param kern: Kernel function of self-similarity matrixes
    :param kerngeno: Kernel function for SNPs marker matrix. If you do not have marker matrix, it should be None.
    :param genomat: SNPs marker matrix: N_G * Markers nd-array matrix
    :param mode: str object which indicates the scenario of data missing. It should be 'unif' or 'fiber'.
    :param r2: Replication number of CV :int
    :param parameter_candidates: Python dictionary object which has indexes; 'gpara_candidates', 'mparag_candidates','pars_candidates','sig2_candidates'. Each index have list of candidates ofr parameters for cross validation.

    :return Python dictionary object which has three indexes; 'result', 'est' and 'Parameters'. 'result' contains competed three-way array data. 'est' contains estimated missing data. 'parameters' contains parameters used for imputation.
    """

    length = prod(nd.shape)
    dim = nd.shape
    candidates = []

    result=copy.deepcopy(nd)
    nd2 = copy.deepcopy(nd)

    best = candidates_searcher(nd2, mode, kern, kerngeno, genomat,  r2, parameter_candidates)
    gpara, mparag, pars, sig2,cv_est,cv_lacked = best

    candidates.append(best)
    vecten2 = nd2.reshape(length, order='F')

    miss_site = np.isnan(vecten2)

    sc = scaling(nd)
    nduse_sc = dt(sc['dtensor'])
    nduse_meansd = sc['mean_sd']
    nduse_impute = mkk.ten_imputes(nduse_sc.__array__())

    kerndic = mkk.selfkern(nduse_impute, pars, pars, pars, kern)
    kerns = [kerndic['env'],kerndic['geno'],kerndic['trait']]

    if genomat is not None:  ##If you do not have genotype marker data.
        gkern = kerngeno(genomat, gpara)
        if mparag > 0:
            kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
        else:
            kerns[1] = gkern

    solver = GPI_normal(kerns, sig2, nduse_impute, miss_site)

    scaled = solver.reshape(dim, order='F')
    est = reconstruct(scaled, nduse_meansd)
    result[np.isnan(nd)]=est[np.isnan(nd)]

    rdic = {
        'result': result,   #Competed three-way array data
        'est': est,     #Estimated missing data
        'Parameters': candidates, #Parameters used for imputation
        'lacked_data_in_cv':cv_lacked,
        'estimated_value_in_cv':cv_est
    }

    return (rdic)
