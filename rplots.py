import os
import numpy as np
import pandas as pd
import sys
sys.path.append('cgtools')
from plotting import plot_mean_sem, plot_heatmaps, HeatMap
from set_bfactors import update_bfactors
from system import CGSystem
from cli import sbatch, run
                
sysdir = 'ribosomes' 
sysnames = [ 'ribosome', 'ribosome_k',] 
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
    fnames = [f for f in os.listdir(os.path.join(sysdir, 'ribosome', 'data')) if f.startswith('rmsf_')]
    figdir = os.path.join(sysdir, 'png')    
    plot_csvs(fnames, figdir)
    
    
def pdb():
    ribosome = 'ribosome_k'
    metric = 'dci_A'
    sysnames = [ribosome, 'ribosome']
    x = []
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        fdir =  os.path.join(sysdir, sysname, 'data')
        fnames = [f for f in os.listdir(fdir) if f.startswith(f'{metric}.')]
        datas = [pd.read_csv(os.path.join(fdir, fname), header=None) for fname in fnames]
        if metric == 'rmsf':
            factor = 10
        if metric == 'dfi':
            factor = 1e4
        factor = 1e4
        data = datas[0] * factor
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
    sysname = 'ribosome_k'
    system = CGSystem(sysdir, sysname)
    fdir =  os.path.join(sysdir, sysname, 'data')
    fnames = [f for f in os.listdir(fdir) if f.startswith('dci_k')]
    datas = [pd.read_csv(os.path.join(fdir, fname), header=None) for fname in fnames]
    inpdb = system.inpdb
    inpdb = os.path.join(system.wdir, 'ref.pdb')
    for data, fname in zip(datas, fnames):
        print(f'Processing {fname}')
        data = data 
        b_factors = data[1]
        # b_factors -= np.average(b_factors)
        errs = data[2]
        pfix = fname.split('.')[0]
        outpdb = os.path.join(sysdir, 'pdb', f'{sysname}_{pfix}.pdb')
        update_bfactors(inpdb, b_factors, outpdb)
        errpdb = os.path.join(sysdir, 'pdb', f'{sysname}_{pfix}_err.pdb')
        update_bfactors(inpdb, errs, errpdb)
        
        
def ch_dci():
    print('STARTING')
    sysname = 'ribosome'
    system = CGSystem(sysdir, sysname)
    fdir =  os.path.join(sysdir, sysname, 'data')
    fnames = [f for f in os.listdir(fdir) if f.startswith('ch_dci')]
    fnames = sorted(fnames)
    files = [os.path.join(fdir, f) for f in fnames]
    figname = 'ch_dci.png'
    figpath = os.path.join(system.pngdir, figname) 
    plot_heatmaps(files, figpath, vmin=0, vmax=5, cmap='Reds', shape=(2, 1))
    

def ch_dci_diff():
    sysnames = ['ribosome', 'ribosome_k']
    datas = []
    errs = []
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        dfile = os.path.join(system.datdir, 'ch_dci.csv')
        efile = os.path.join(system.datdir, 'ch_dci_err.csv')
        data = pd.read_csv(dfile, sep=',', header=None) 
        err = pd.read_csv(efile, sep=',', header=None)
        datas.append(data)
        errs.append(err)
    data = datas[1] - datas[0]
    errs= np.sqrt(errs[1]**2 + errs[0]**2)
    figname = 'ch_ddci.png'
    figpath = os.path.join(sysdir, 'png', figname) 
    datas = [[data], [errs]]
    labels = [['dDCI'], ['Error']]
    plot = HeatMap(datas, labels, vmin=-0.25, vmax=0.25, cmap='bwr', shape=(2, 1))
    plot.save_figure(figpath) 
   
    
if __name__ == '__main__':
    # pdb()
    # png()
    dci_pdb()
    # ch_dci()
    # ch_dci_diff()

