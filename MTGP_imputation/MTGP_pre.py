import numpy as np
from sktensor import dtensor as dt
import copy
import itertools as it
import os
import mkkernel as mkk

def prod(seq):
    prod=1
    for element in seq:
        prod=prod*seq
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

def kouho_searcher(nd, mode, *args):
    """

    :param nd: three mode tenosr array data
    :param mode: str object which indicates the scenario of data missing.
    :param args: Set of parameters
    :return: test error of CV
    """

    global kouho
    kern, kerngeno, genomat, r2, parameter_kouho = args
    kouhopre = sorted(parameter_kouho.items())
    bestuse = [kouhopre[i][1] for i in range(len(kouhopre))]

    envpara_kouho, gpara_kouho,mparae_kouho,  mparag_kouho, pars_kouho, sig2_kouho = bestuse
    nd_l = copy.deepcopy(nd)
    dim = nd_l.shape
    dim2 = list(copy.deepcopy(dim))
    dim2.append(r2)
    length = prod(dim)

    mse_top = np.array(1000000.)
    if mode == 'inner':
        nk = prod(dim)
        print("Inner")

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
    elif mode == 'inner_fiber':
        nk = dim[0]*dim[1]
        print("Inner_fiber")

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

    # for npars in range(len(pars_kouho)):
    #     for ngpara in range(len(gpara_kouho)):
    #         for nepara in range(len(envpara_kouho)):
    #             for nmparag in range(len(mparag_kouho)):
    #                 for nmparae in range(len(mparae_kouho)):
    #                     for nsig2 in range(len(sig2_kouho)):
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
            nd2 = copy.deepcopy(nd_l)

            if mode == 'inner':
                u = stock[p]
                vectenx = nd2.reshape(length, order='F')
                vectenx[u] = np.nan
                nd2 = vectenx.reshape(dim, order='F')

            elif mode == 'inner_fiber':
                u = stock[p]
                vectenx = nd2.reshape([dim[0]*dim[1],dim[2]], order='F')
                vectenx[u,:] = np.nan
                nd2 = vectenx.reshape(dim, order='F')

            vecten2 = nd2.reshape(length, order='F')
            full_site = np.logical_not(np.isnan(vecten2))
            pre = vecten2[full_site]
            vecten2[full_site] = pre
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

            if genomat.any():
                gkern = kerngeno(genomat, gpara)
                kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
            solver = GPI_normal_true2(kerns, sig2, nduse_impute, miss_site)

            origin[:, :, :, p] = nd
            nd_compare = compare(nd, nduse_meansd)
            comp[:, :, :, p] = nd_compare
            scaled[:, :, :, p] = solver.reshape(dim, order='F')
            est[:, :, :, p] = reconstruct(scaled[:, :, :, p], nduse_meansd)
        mse = np.nanmean((scaled - comp) ** 2) ** 0.5

        if mse < mse_top:
            kouho = [pars, gpara, mparag, sig2]
            mse_top = copy.deepcopy(mse)
    print(mse_top)

    parsf = kouho[0]
    gparaf = kouho[1]
    envparaf = kouho[2]
    mparaf = kouho[3]
    emparaf = kouho[4]
    sig2f = kouho[5]

    assert isinstance(sig2f, object)
    return envparaf,gparaf, emparaf,mparaf,parsf, sig2f

def MTGP_impute(nd, kern, name, kerngeno, genomat, kernenv, envmat, mode, r2,
                                        parameter_kouho, miss_rate, randomseed):
    """

    :param nd: N_E*N_G*N_T three-way nd-array data: N_E environments, N_G genotypes and N_T traits.
    :param kern: Kernel function for self-similarity matrix
    :param name: Name of result-folder :str
    :param kerngeno: Kernel function for SNPs marker matrix
    :param genomat: SNPs marker matrix: N_G * Markers nd-array matrix
    :param mode: How to make missing data for cross validation (CV). "Inner" or "Inner-fiber"
    :param r2: Replication number of CV :int
    :param parameter_kouho:
    :param randomseed:
    :return:
    """


    length = prod(nd.shape)
    dim = nd.shape
    kouho = []

    rx = 1
    np.random.seed(50 * randomseed)

    nd_miss = copy.deepcopy(nd)  ##11/02修正　11/05改造

    scaled = np.tile([np.nan], dim)
    est = np.tile([np.nan], (dim))
    origin = np.tile([np.nan], (dim))
    comp = np.tile([np.nan], (dim))

    count = -1

        count += 1
        if count/5==0:
            print(count)
        print(str(count)[:2] + '_Main')
        nd2 = copy.deepcopy(nd_miss)

        best = kouho_searcher(nd2, mode, kern, kerngeno, genomat, kernenv, envmat, r2, parameter_kouho)
        envpara, gpara, mparag, mparae, pars, sig2 = best

        kouho.append(best)  ##parstを追加
        vecten2 = nd2.reshape(length, order='F')
        full_site = np.logical_not(np.isnan(vecten2))
        pre = vecten2[full_site]
        vecten2[full_site] = pre
        miss_site = np.isnan(vecten2)
        nduse = vecten2.reshape(dim, order='F')

        sc = scaling(nduse)
        nduse_sc = dt(sc['dtensor'])
        nduse_meansd = sc['mean_sd']
        nduse_impute = dt(mkk.ten_imputes(nduse_sc.__array__()))
        kerns = list()
        kerndic = mkk.selfkern(nduse_impute, pars, pars, pars, kern)
        kerns.append(kerndic['env'])
        kerns.append(kerndic['geno'])
        kerns.append(kerndic['trait'])

        if genomat.any():  ##系統の付加情報
            gkern = kerngeno(genomat, gpara)
            if mparag > 0:
                kerns[1] = mparag * (kerns[1]) + (1 - mparag) * gkern
            else:
                kerns[1] = gkern

        solver = GPI_normal_true2(kerns, sig2, nduse_impute, miss_site)
        origin[:, :, :, count] = nd
        nd_compare = compare(nd, nduse_meansd)
        comp[:, :, :, count] = nd_compare

        scaled[:, :, :, count] = solver.reshape(dim, order='F')
        est[:, :, :, count] = reconstruct(scaled[:, :, :, count], nduse_meansd)

    rdic = {}
    rdic['scaled'] = scaled
    rdic['est'] = est
    rdic['origin'] = origin
    rdic['comp'] = comp
    rdic['miss'] = nd_miss

    rdic['kouhos'] = kouho
    #########make directory and save######
    if not (os.path.exists('pidata/' + name)):
        os.mkdir('pidata/' + name)
    dirname = 'pidata/' + name + '/'
    n1 = dirname + '_scaled' + '.txt'
    np.savetxt(n1, scaled.reshape(prod(dim2), order='F'))
    n2 = dirname + '_est' + '.txt'
    np.savetxt(n2, est.reshape(prod(dim2), order='F'))
    n3 = dirname + '_origin' + '.txt'
    np.savetxt(n3, origin.reshape(prod(dim2), order='F'))
    n4 = dirname + '_comp' + '.txt'
    np.savetxt(n4, comp.reshape(prod(dim2), order='F'))
    n5 = dirname + '_miss' + '.txt'
    np.savetxt(n5, nd_miss.reshape(prod(dim), order='F'))

    f = file(dirname + '/' + 'res.dump', 'w')  ##2015/9/1
    pk.dump(rdic, f)
    f.close()

    times = time.time() - start
    rdic['time'] = times
    f = file(dirname + '/' + 'res.dump', 'w')  ##2015/9/1
    pk.dump(rdic, f)
    f.close()

    return (rdic)
