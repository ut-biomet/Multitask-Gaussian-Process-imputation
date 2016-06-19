# -*- coding: utf-8 -*-

#2016/5/23 Functions for making tensor array data.#

import numpy as np
import pandas as pd
import sys

def make_tensor_data(list_filenames):
    """

    :param list_filenames:List of file names that composes of tensor data. All the files should be csv files with the size of N_E environments x N_G environments. The length of the list should be N_T
    :return: nd array object with the size of:N_E x N_G x N_T
    """

    list_files=[pd.read_csv(filename).ix[:,1:].values for filename in list_filenames ]

    list_shapes=[csv.shape for csv in list_files ]
    shape_pre=list(set(list_shapes))
    if len(shape_pre)!=1:
        print("Size of data is different")
        sys.exit()
    shape=shape_pre[0]
    tshape=list(shape)+[len(list_filenames)]

    tensor=np.tile(np.nan,tshape)
    for i, data in enumerate(list_files):
        tensor[:,:,i]=data

    return(tensor)

