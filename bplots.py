import os
import numpy as np
import pandas as pd
import sys
sys.path.append('cgtools')
from plotting import plot_mean_sem, plot_xvg
from set_bfactors import update_bfactors
from system import CGSystem
from cli import sbatch, run
                
sysdir = 'systems' 
sysnames = ['1btl_meta', '1btl', ] 


def pull_sys_files(sysnames, fdir, fname):
    systems = [CGSystem(sysdir, sysname) for sysname in sysnames]
    fpaths = [os.path.join(system.wdir, fdir, fname) for system in systems]
    fpaths = [f for f in fpaths if os.path.exists(f)]
    return fpaths


def pull_run_files(system, fdir, fname):
    runnames = system.mdruns
    runs = [system.initmd(runname) for runname in runnames]
    fpaths = [os.path.join(run.rundir, fdir, fname) for run in runs]
    fpaths = [f for f in fpaths if os.path.exists(f)]
    return fpaths
        
        
def plot_csvs(fnames, figdir):
    for fname in fnames:
        figpath = os.path.join(figdir, fname.replace('csv', 'png'))
        files = pull_files(sysnames, 'data', fname)
        plot_mean_sem(files, figpath, )
        
        
def plot_systems(): 
    fnames = [f for f in os.listdir(os.path.join(sysdir, sysnames[0], 'data')) if f.startswith('r')]
    figdir = os.path.join(sysdir, 'png')    
    plot_csvs(fnames, figdir)
    

def plot_runs(fdir, fname):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname) 
        files = pull_run_files(system, fdir, fname)
        figpath = os.path.join(system.pngdir, fname.replace('xvg', 'png'))
        plot_xvg(files, figpath) 
    

if __name__ == '__main__':
    plot_runs('rms_analysis', 'rmsf.xvg')

