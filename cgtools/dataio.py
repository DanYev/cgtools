import os
import numpy as np
import pandas as pd


def read_data(fpath):
    """ 
    Reads a .csv or .npy file 
    
    Input 
    ------
    fname: string
        Name of the file to read
    Output 
    ------
    data: ndarray
        Numpy array
    """
    ftype = fpath.split('.')[-1]
    if ftype == 'npy':
        try:
            data = np.load(fpath)
        except:
            raise ValueError()
    if ftype == 'csv' or ftype == 'dat':
        try:
            df = pd.read_csv(fpath, sep='\\s+', header=None)
            data = df.values
            data = np.squeeze(df.values)
            if data.shape[0] != 1104:
                raise ValueError()
        except:
            raise ValueError()
    if ftype == 'xvg':
        try:
            df = pd.read_csv(fpath, sep='\\s+', header=None, usecols=[1])
            data = df.values
            data = np.squeeze(df.values)
            if data.shape[0] > 10000:
                raise ValueError()
        except:
            raise ValueError()
    return data
    
    
def read_xvg(fpath, usecols=[0, 1]):
    """ 
    Reads a GROMACS xvg file
    
    Input 
    ------
    fname: string
        Name of the file to read
    Output 
    ------
    data: ndarray
        Numpy array
    """
    try:
        df = pd.read_csv(fpath, sep='\\s+', header=None, usecols=usecols)
        data = df
    except:
        raise ValueError()
    return data
    
    
def save_data(data, fpath):
    """ 
    Saves the data as a .csv or .npy file 
    
    Input 
    ------
    data: numpy array
        Numpy array of data
    fpath: string
        Path to the file to save
    Output 
    ------
    """
    ftype = fpath.split('.')[-1]
    if ftype == 'npy':
        np.save(fpath, data)
    if ftype == 'csv':
        df = pd.DataFrame(data)
        df.to_csv(fpath, index=False, header=None, float_format='%.3E', sep=',')
        
        
def calc_mean_sem(datas):
    datas = np.array(datas)
    mean = np.average(datas, axis=0)
    sem = np.std(datas, axis=0) / np.sqrt(datas.shape[0])
    return mean, sem
        