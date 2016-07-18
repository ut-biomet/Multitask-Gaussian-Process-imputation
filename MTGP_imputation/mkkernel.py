# -*- coding: utf-8 -*-
#2016/5/23 Functions for making tensor array data.#

import numpy as np
from sktensor import dtensor as dt
import MTGP as MG
from scipy.spatial import distance as dist
def linear_kernel(x1,par=None):
    """
    Function to make linear kernel of matrix.
    :param x1: nd array object with n1*p size

    :param par: Not important value.
    :rtype ndarray
    :return:Linear kernel with size n1*n1
    """
    prod=x1.dot(x1.T)
    return(prod)

def Gauss_kernel(x1,par):
    """

    :param ndarray x1:n1*p size matrix
    :type par: float
    :param par: kernel band width
    :rtype: n1*n2 ndarray
    :return:: Gaussian kernel matrix
    """

    distmat=dist.squareform(dist.pdist(x1))
    kernmat=np.exp(-0.5*distmat/par)
    return(kernmat)

def median_RBF_kernel(x1,par):
    """
    Function to calculate RBF kernel standardized by median of squared norm between vectors in matrix.
    :param ndarray x1:n1*p size matrix
    :type par: float
    :param par: kernel band width parameter.
    :rtype: ndarray
    :return:: Gaussian kernel matrix with the size of n1*n1
    """

    distvec=dist.pdist(x1)
    distmat=dist.squareform(distvec)
    kernmat=np.exp(-par*distmat/np.median(distvec))
    return(kernmat)


def ten_imputes(nduse_sc):
    """
    Function to impute missing data temporarily with mean of each environment, genotypes at each trait.
    :type nduse_sc: three-way nd array with the size of:N_E x N_G x N_T
    :param nduse_sc: Scaled missing phenotypic values.
    :type ndarray
    :return: Temporarily imputed phenotypic values with the size of N_E x N_G x N_T
    """
    dim=nduse_sc.shape
    ten_imputed=np.zeros(dim)
    ten_imputed[np.logical_not(np.isnan(nduse_sc))]=nduse_sc[np.logical_not(np.isnan(nduse_sc))]
    chart_e=np.zeros((dim[0],dim[2]))
    mean_e=np.zeros((dim[0],dim[2]))
    chart_g=np.zeros((dim[1],dim[2]))
    mean_g=np.zeros((dim[1],dim[2]))
    vecten=nduse_sc.reshape(MG.prod(dim),order='F')
    k=np.array(range(MG.prod(dim))).astype(np.int)[np.isnan(vecten)]
    z=k//(dim[0]*dim[1])
    y=(k-z*dim[0]*dim[1])//(dim[0])
    x=(k-z*(dim[0]*dim[1])-y*dim[0])
    for i in range(len(k)):
        if chart_e[x[i],z[i]]==0:
            mean_x=np.nanmean(nduse_sc[x[i],:,z[i]])
            mean_e[x[i],z[i]]=mean_x
            chart_e[x[i],z[i]]+=1
            if mean_x!=mean_x:
                mean_e[x[i],z[i]]=0
        if chart_g[y[i],z[i]]==0:
            mean_y=np.nanmean(nduse_sc[:,y[i],z[i]])
            mean_g[y[i],z[i]]=mean_y
            if mean_y!=mean_y:
                mean_g[y[i],z[i]]=0
            chart_g[y[i],z[i]]+=1
        ten_imputed[x[i],y[i],z[i]]=mean_e[x[i],z[i]]+mean_g[y[i],z[i]]
    return(ten_imputed)

def selfkern(nd, thetag_self, thetae_self, thetat_self, self_kern):
    """
    Function to impute missing tensor data with mean values. Missing data are temporary imputed by the deviancec of each genotype and,each environment .
    :type nd: ndarray
    :param nd: Ne*Ng*Nt three-way nd-array data: Ne environments, Ng genotypes and Nt traits.
    :param self_kern: Kernel function for self-similarity matrix
    :type thetag_self: float
    :param thetag_self: Parameter of kernel function for kernel matrix of genotype (which corresponds to lambda in the paper).
    :type thetae_self: float
    :param thetae_self: Parameter of kernel function for kernel matrix of environment.
    :type thetat_self: float
    :param thetat_self: Parameter of kernel function for kernel matrix of traits.
    :rtype: dict
    :return:Python dictionary object which contatins three self measuring similarity kernel of environment, genotype and trait.
    """

    # dim = nd.shape
    nduse_impute = dt(ten_imputes(nd))
    kerns = {}
    unfold0 = nduse_impute.unfold(0)
    kerns['env'] = (self_kern(unfold0.__array__(), thetae_self))
    unfold1 = nduse_impute.unfold(1)
    kerns['geno'] = (self_kern(unfold1.__array__(), thetag_self))
    unfold1 = nduse_impute.unfold(2)
    kerns['trait'] = (self_kern(unfold1.__array__(), thetat_self))

    return (kerns)
