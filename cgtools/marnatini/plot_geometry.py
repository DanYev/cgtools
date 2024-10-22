import os
import numpy as np
import matplotlib.pyplot as plt


def plot_geometry(bonds, angles, dihedrals, cgbonds, cgangles, cgdihedrals, topology, molecule, resname):
    
    bonds = np.array(bonds).T * 10.0
    angles = np.array(angles).T
    dihedrals = np.array(dihedrals).T
    cgbonds = np.array(cgbonds).T * 10.0
    cgangles = np.array(cgangles).T
    cgdihedrals = np.array(cgdihedrals).T
    
    out = os.path.join('png', f'{molecule}')
    if not os.path.isdir(out):
        os.mkdir(out)
        
    n_bins = 20    
    aakwargs = {'bins': n_bins, 'density': True, 'fill': True,}
    cgkwargs = {'bins': n_bins, 'density': True, 'fill': False, 'rwidth': 0.8,}

    # Plotting bonds    
    fig, axs = plt.subplots(2, 4, sharey=False, figsize=(12,6))
    arrout = []
    for n, key in enumerate(topology['bonds'].keys()):
        i = n // 4
        j = n % 4
        avg = np.average(bonds[n])
        std = np.std(bonds[n])
        cgavg = np.average(cgbonds[n])
        cgstd = np.std(cgbonds[n])
        axs[i, j].hist(bonds[n], **aakwargs)
        axs[i, j].hist(cgbonds[n], **cgkwargs)
        axs[i, j].title.set_text(f'Bond {key}\nAA: AVG:{avg:2.2f}, STD:{std:2.2f}\nCG: AVG:{cgavg:2.2f}, STD:{cgstd:2.2f}')
        arrout.append(0.1 * avg)
    fig.suptitle(f'bonds: {resname}', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{out}/bonds_{resname}.png')
    plt.close()
    
    # Plotting angles         
    fig, axs = plt.subplots(2, 5, sharey=False, figsize=(12.8,6))
    arrout = []
    for n, key in enumerate(topology['angles'].keys()):
        i = n // 5
        j = n % 5
        avg = np.average(angles[n])
        std = np.std(angles[n])
        cgavg = np.average(cgangles[n])
        cgstd = np.std(cgangles[n])
        axs[i, j].hist(angles[n], **aakwargs)
        axs[i, j].hist(cgangles[n], **cgkwargs)
        axs[i, j].title.set_text(f'Angle {key}\nAA: AVG:{avg:2.2f}, STD:{std:2.2f}\nCG: AVG:{cgavg:2.2f}, STD:{cgstd:2.2f}')
        arrout.append(avg)
    fig.suptitle(f'Angles: {resname}', fontsize=16)    
    fig.tight_layout()
    fig.savefig(f'{out}/angles_{resname}.png')  
    plt.close()
    
    # Plotting dihedrals
    fig, axs = plt.subplots(2, 4, sharey=False, figsize=(12,6))
    arrout = []
    for n, key in enumerate(topology['dihedrals'].keys()):
        i = n // 4
        j = n % 4
        avg = np.average(dihedrals[n])
        std = np.std(dihedrals[n])
        cgavg = np.average(cgdihedrals[n])
        cgstd = np.std(cgdihedrals[n])
        axs[i, j].hist(dihedrals[n], **aakwargs)
        axs[i, j].hist(cgdihedrals[n], **cgkwargs)
        axs[i, j].title.set_text(f'Dihedral {key}\nAA: AVG:{avg:2.2f}, STD:{std:2.2f}\nCG: AVG:{cgavg:2.2f}, STD:{cgstd:2.2f}')
        arrout.append(avg)
    fig.suptitle(f'Dihedrals: {resname}', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{out}/dihedrals_{resname}.png')
    plt.close()
    