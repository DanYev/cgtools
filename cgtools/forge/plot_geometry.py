import os
import numpy as np
import matplotlib.pyplot as plt


def save_data(bonds, angles, dihedrals, cgbonds, cgangles, cgdihedrals,):
    bonds = np.array(bonds)
    angles = np.array(angles)
    dihedrals = np.array(dihedrals)
    cgbonds = np.array(cgbonds)
    cgangles = np.array(cgangles)
    cgdihedrals = np.array(cgdihedrals)
    np.save('data/bonds.npy', bonds)
    np.save('data/angles.npy', angles)
    np.save('data/dihedrals.npy', dihedrals)
    np.save('data/cgbonds.npy', cgbonds)
    np.save('data/cgangles.npy', cgangles)
    np.save('data/cgdihedrals.npy', cgdihedrals)
    
    
def load_data():
    bonds = np.load('data/bonds.npy').T
    angles = np.load('data/angles.npy').T
    dihedrals = np.load('data/dihedrals.npy').T
    cgbonds = np.load('data/cgbonds.npy').T
    cgangles = np.load('data/cgangles.npy').T
    cgdihedrals = np.load('data/cgdihedrals.npy').T
    return bonds, angles, dihedrals, cgbonds, cgangles, cgdihedrals


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
    fig, axs = plt.subplots(4, 4, sharey=False, figsize=(12,8))
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
    fig, axs = plt.subplots(3, 4, sharey=False, figsize=(12,8))
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
    
    
def make_fig_bonded():
    bonds, angles, dihedrals, cgbonds, cgangles, cgdihedrals = load_data()
    bonds = np.stack((bonds[0, :], bonds[1, :], )).flat
    cgbonds = np.stack((cgbonds[0, :], cgbonds[1, :], )).flat
    angles = np.stack((angles[0, :], angles[1, :], )).flat
    angles = angles[angles < 160]
    cgangles = np.stack((cgangles[0, :], cgangles[1, :], )).flat
    dihedrals = np.stack((dihedrals[0, :], dihedrals[1, :], )).flat
    dihedrals = dihedrals[dihedrals > -100]
    dihedrals = dihedrals[dihedrals < 150]
    cgdihedrals = np.stack((cgdihedrals[0, :], cgdihedrals[1, :], )).flat
    
    out = os.path.join('png', 'figs')
    if not os.path.isdir(out):
        os.mkdir(out)
        
    n_bins = 500 
    aakwargs = {'label':'All-atom', 'bins': n_bins, 'density': True, 'histtype':'step', 'lw':3}
    cgkwargs = {'label':'Coarse-grained', 'bins': n_bins, 'density': True, 'histtype':'step', 'lw':3}

    # Plotting bonds    
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.hist(bonds, **aakwargs)
    plt.hist(cgbonds, **cgkwargs)
    plt.autoscale(tight=True)
    plt.title('Backbone Bonds', fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=18)
    plt.xlabel('Distance (nm)', fontsize=20)
    plt.xlim([0.25, 0.50])
    plt.figtext(0.58, 0.68 , 'BB1-BB2 and BB2-BB1', fontsize=18, )
    plt.legend(frameon=False, fontsize=18, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{out}/fig_2_bonds.png')
    plt.close()
    # Plotting angles    
    fig = plt.figure(figsize=(8.0, 5.0))
    plt.hist(angles, **aakwargs)
    plt.hist(cgangles, **cgkwargs)
    plt.autoscale(tight=True)
    plt.title('Backbone Angles', fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=18)
    plt.xlabel('Angle (degrees)', fontsize=20)
    # plt.legend(frameon=False, fontsize=18, loc='lower right')
    plt.figtext(0.67, 0.77 , 'BB1-BB2-BB1\nBB2-BB1-BB2', fontsize=18, )
    fig.tight_layout()
    fig.savefig(f'{out}/fig_2_angles.png')
    plt.close()
    # Plotting dihedrals    
    fig = plt.figure(figsize=(8.0, 5.0))
    plt.hist(dihedrals, **aakwargs)
    plt.hist(cgdihedrals, **cgkwargs)
    plt.autoscale(tight=True)
    plt.title('Backbone Dihedrals', fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=18)
    plt.xlabel('Angle (degrees)', fontsize=20)
    # plt.legend(frameon=False, fontsize=18, loc='lower right')
    plt.figtext(0.62, 0.77 , 'BB1-BB2-BB1-BB2\nBB2-BB1-BB2-BB1', fontsize=18, )
    fig.tight_layout()
    fig.savefig(f'{out}/fig_2_dih.png')
    plt.close()

    