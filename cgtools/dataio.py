import os
import numpy as np
import pandas as pd
from pathlib import Path


def fname_filter(f, sw='', cont='', ew=''):  
    """
    Filters a file name based on its start, substring, and end patterns.
    """
    return f.startswith(sw) and cont in f and f.endswith(ew)
    

def filter_files(fpaths, sw='', cont='', ew=''):  
    """
    Filters files in a list using the above filter
    """
    files = [f for f in fpaths if fname_filter(f.name, sw=sw, cont=cont, ew=ew)]
    return files
    

def pull_all_files(directory, ):
    """
    Recursively lists all files in the given directory and its subdirectories.
    Parameters: directory (str or Path): The root directory to start searching for files.
    Returns: list[Path]: A list of Path objects, each representing the absolute path
            to a file within the directory and its subdirectories.
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(Path(os.path.join(root, file)))
    return all_files  


def pull_prefix_files(directory, prefix):
    """
    Recursively lists files that start with 'prefix' in the given directory and its subdirectories.
    Parameters: directory (str or Path): The root directory to start searching for files.
    Returns: list[Path]: A list of Path objects, each representing the absolute path
            to a file within the directory and its subdirectories.
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                all_files.append(Path(os.path.join(root, file)))
    return all_files  


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
        