import os
import numpy as np
import pandas as pd
import sys
sys.path.append('cgtools')
from plotting import plot_mean_sem
from set_bfactors import update_bfactors
from system import CGSystem
from cli import sbatch, run
                
sysdir = 'ribosomes' 
# sysnames = ['ribosome', 'ribosome_k', ] 
# sysnames = ['ribosome_aa']


def pull_files(sysnames, fdir, fname):
    systems = [CGSystem(sysdir, sysname) for sysname in sysnames]
    fpaths = [os.path.join(system.wdir, fdir, fname) for system in systems]
    fpaths = [f for f in fpaths if os.path.exists(f)]
    return fpaths
        
        
def plot_csvs(fnames, figdir):
    for fname in fnames:
        figpath = os.path.join(figdir, fname.replace('csv', 'png'))
        files = pull_files(sysnames, 'data', fname)
        plot_mean_sem(files, figpath, )
    
    
def png(): 
    fnames = [f for f in os.listdir(os.path.join(sysdir, 'ribosome', 'data')) if f.startswith('r')]
    figdir = os.path.join(sysdir, 'png')    
    plot_csvs(fnames, figdir)
    
    
def pdb():
    ribosome = 'ribosome_k'
    metric = 'dfi'
    sysnames = [ribosome, 'ribosome']
    x = []
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        fdir =  os.path.join(sysdir, sysname, 'data')
        fnames = [f for f in os.listdir(fdir) if f.startswith(f'{metric}.')]
        datas = [pd.read_csv(os.path.join(fdir, fname), header=None) for fname in fnames]
        data = datas[0] * 1e4
        x.append((data[1], data[2]))
    
    b_factors = (x[0][0] - x[1][0]) * 10
    # b_factors = (x[0][0]) * 10
    errs = np.sqrt(x[0][1]**2 + x[1][1]**2) * 10
    sysname = ribosome
    inpdb = os.path.join(system.inpdb)
    # CGSystem.mask_pdb(system.syspdb, inpdb, mask=['BB', 'BB2'])
    outpdb = os.path.join(sysdir, 'pdb', f'{sysname}_d{metric}.pdb')
    update_bfactors(inpdb, b_factors, outpdb)
    errpdb = os.path.join(sysdir, 'pdb', f'{sysname}_d{metric}_err.pdb')
    update_bfactors(inpdb, errs, errpdb)
    
    
def dci_pdb():
    sysname = 'ribosome'
    system = CGSystem(sysdir, sysname)
    fdir =  os.path.join(sysdir, sysname, 'data')
    fnames = [f for f in os.listdir(fdir) if f.startswith('dfi')]
    datas = [pd.read_csv(os.path.join(fdir, fname), header=None) for fname in fnames]
    inpdb = system.inpdb
    inpdb = os.path.join(system.wdir, 'ref.pdb')
    for data, fname in zip(datas, fnames):
        print(f'Processing {fname}')
        data = data * 1e4
        b_factors = data[1] 
        errs = data[2]
        pfix = fname.split('.')[0]
        outpdb = os.path.join(sysdir, 'pdb', f'{sysname}_{pfix}.pdb')
        update_bfactors(inpdb, b_factors, outpdb)
        errpdb = os.path.join(sysdir, 'pdb', f'{sysname}_{pfix}_err.pdb')
        update_bfactors(inpdb, errs, errpdb)
   
    
if __name__ == '__main__':
    # pdb()
    # png()
    dci_pdb()

