import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

system = sys.argv[1]

def percentile(x):
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px


def set_plotting_parameters(name='xvg', xlabel=None, ylabel=None, legend=False, loc='lower right'):
    if legend:
        plt.legend(frameon=False, fontsize=13, loc=loc)
    plt.autoscale(tight=True)
    plt.title(f'{name}', fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid()
    plt.tight_layout()
    
    
def initialize(directory, observable, save_to):
    xlabels = {'rmsf': 'residue', 'rmsd': 'time', 'rg': 'time'}
    xlabel = xlabels[observable]
    path_to_xvgs = f'analysis/{directory}'
    if not os.path.exists(f'{save_to}/{directory}'):
        os.makedirs(f'{save_to}/{directory}')
    return xlabel, path_to_xvgs
    

def read_csv(filepath, sep='\\s+', header=None, usecols=[0, 1]):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, sep=sep, header=header, usecols=usecols)
        x = np.array(df[df.iloc[:, usecols[0]].notna()].iloc[:, usecols[0]])
        y = np.array(df[df.iloc[:, usecols[1]].notna()].iloc[:, usecols[1]])
        x.resize(y.size)
    else:
        x = np.nan
        y = np.nan
    return x, y


def plot_xvg(x, y, label='avg', avg=False, color=None, alpha=1.0):
    if avg:
        plt.plot(x, y, color='gray', alpha=alpha)
    else:
        plt.plot(x, y, color=color, alpha=alpha, label=label)


def plot_xvgs(path_to_xvgs, xvg_files, avg, alpha):
    average = []
    x = np.nan
    y_mean = np.nan
    for file in xvg_files:
        x, y = read_csv(path_to_xvgs, file)
        if not avg:
            plot_xvg(x, y, file, avg, alpha)
            print(np.average(y))
        else:
            average.append(y)
    if avg:
        y_mean = np.average(np.asarray(average, float), axis=0)
        plt.plot(x, y_mean)
    return y_mean
    

def plot_rmsf(files=[]):
    for file in files:
        tag = file.split('.')[0].split('/')[-1][5:]
        fig = plt.figure(figsize=(12.8, 5.0))
        x, y = read_csv(file)
        x = np.arange(0, len(y))
        y *= 10.0
        y_av = np.average(y)
        plot_xvg(x, y, label=tag)
        set_plotting_parameters(name=f'{system}_{tag}, av={y_av:.3f}', xlabel='Res', ylabel='RMSF, A', legend=True, loc='upper right')
        # plt.ylim(bottom=0, top=5.0)
        fig.savefig(f'png/{system}/rmsf_{tag}.png')
        np.save(f'data/{system}/rmsf_{tag}.npy', y)
        plt.close()
        

def plot_dfi(files=[], do_perc=False, avg=False, alpha=1.0):
    fig = plt.figure(figsize=(12.8, 5.0))
    ys = []
    for file in files:
        x, y = read_csv(file)
        if do_perc:
            y = percentile(y)
        plot_xvg(x, y, avg=avg, alpha=alpha, label=f'{file[-5:-4]}')
        ys.append(y)
    y_av = np.average(ys, axis=0)    
    if len(ys) > 1:
        plot_xvg(x, y_av, color='k')
    if not do_perc:
        set_plotting_parameters(name=f'{system}', xlabel='Res', ylabel='DFI', legend=True, loc='upper right')
        plt.ylim(bottom=0, top=300)
        fig.savefig(f'png/{system}_dfi.png')
        np.save(f'data/{system}_dfi.npy', y_av/np.sum(y_av))
    else:
        set_plotting_parameters(name=f'{system}', xlabel='Res', ylabel='%DFI', legend=True, loc='lower left')
        fig.savefig(f'png/{system}_pdfi.png')
    plt.close()
 
 
def plot_rdf(files=[]):
    fig = plt.figure(figsize=(12.8, 5.0))
    for file in files:
        tag = file.split('.')[0].split('/')[-1][4:]
        x, y = read_csv(file)
        plot_xvg(x, y, label=tag)
        set_plotting_parameters(name=f'{system}', xlabel='R, nm', ylabel='RDF', legend=True, loc='upper right')
    fig.savefig(f'png/{system}/rdf.png')
    plt.close()   
    
    
def plot_rmsd(files=[], avg=False, alpha=1.0):
    fig = plt.figure(figsize=(12.8, 5.0))
    ys = []
    for file in files:
        tag = file.split('.')[0].split('/')[-1][5:]
        x, y = read_csv(file)
        x *= 1e-3
        y *= 10.0
        plot_xvg(x, y, avg=avg, alpha=alpha, label=f'{file[-5:-4]}')
        ys.append(y)
    y_av = np.average(ys, axis=0)    
    if len(ys) > 1:
        plot_xvg(x, y_av, color='k')
    set_plotting_parameters(name=f'{system}', xlabel='Time, ns', ylabel='RMSD, A', legend=True)
    fig.savefig(f'png/{system}/rmsd_{tag}.png')
    plt.close()  
    

def plot_rmsf_chain(files=[], chain='', do_percent=False):
    fig = plt.figure(figsize=(16.0, 6.0))
    colors = ['gray', 'gray', 'red']
    alphas = [1.0, 1.0, 0.7]
    ys = []
    for file, color, alpha in zip(files, colors, alphas):
        if os.path.exists(file):
            x, y = read_csv(file)
            x = np.arange(len(y))
            if do_percent:
                y = percentile(y)
            plot_xvg(x, y, color=color, alpha=alpha)
            ys.append(y)
    r01, p01 = pearsonr(ys[0], ys[1])
    r01 = np.sqrt(r01)
    if len(ys) == 3:
        r02, p02 = pearsonr(ys[0], ys[2])
        r12, p12 = pearsonr(ys[1], ys[2])
        r02 = np.sqrt(r02)
        r12 = np.sqrt(r12)
    else:
        r02 = 0
        r12 = 0
    set_plotting_parameters(name=f'{chain}_RMSF, {r01:.2f}, {r02:.2f}, {r12:.2f}', xlabel='res', ylabel='rmsf')
    print(f'{chain}_RMSF, {r01:.2f}, {r02:.2f}, {r12:.2f}')
    fig.savefig(f'png/{system}/chains/{chain}_rmsf.png')
    plt.close()
        

def main():
    if not os.path.isdir(f'png/{system}/chains'):
        os.mkdir(f'png/{system}/chains')
    if not os.path.isdir(f'data/{system}'):
        os.mkdir(f'data/{system}')
        
    wdir = f'systems/{system}/analysis'    
    # RMSF
    rmsf_files = [os.path.join(wdir, file) for file in os.listdir(wdir) if file.startswith('rmsf')]
    if rmsf_files:
        plot_rmsf(rmsf_files)
    # RMSD
    rmsd_files = [os.path.join(wdir, file) for file in os.listdir(wdir) if file.startswith('rmsd')]
    if rmsd_files:
        plot_rmsd(rmsd_files)
    # RDF
    rdf_files = [os.path.join(wdir, file) for file in os.listdir(wdir) if file.startswith('rdf')]
    if rdf_files:
        plot_rdf(rdf_files)

    # CHAIN RMSF
    rmsf_files = sorted(os.listdir(f'systems/{system}/rmsf')) # os.listdir(f'systems/{system}/rmsf')
    for file in rmsf_files:
        chain = file[:-9]
        files = [f'systems/all_atom/rmsf/{chain}_rmsf_1.xvg',
                f'systems/all_atom/rmsf/{chain}_rmsf_2.xvg', 
                f'systems/{system}/rmsf/{chain}_rmsf.xvg']
        if os.path.exists(files[0]) or os.path.exists(file[1]):
            plot_rmsf_chain(files, chain, do_percent=True)

    # 
    

if __name__ == '__main__':
    main()

    
    

  
