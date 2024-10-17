import argparse
import importlib.resources
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import subprocess as sp
from pathlib import Path
from pyrotini.get_go import get_go
from . import PRT_DATA, PRT_DICT


def prt_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs CGMD simulation for pyRosetta")
    parser.add_argument("-f", "--pdb", required=True, help="PDB file")
    parser.add_argument("-d", "--wdir", required=True, help="Relative path to the working directory")
    parser.add_argument("-n", "--ncpus", required=False, help="Number of requested CPUS for a batch job")
    return parser.parse_args()

                    
def make_topology_file(wdir, protein='protein'):
    r"""
    -protein        Name of the protein (just for the reference, doesn't affect anything)
    
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    with open('system.top', 'w') as out_file:
        out_file.write(f'#define GO_VIRT\n')
        out_file.write(f'#include "martini.itp"\n')
        out_file.write(f'#include "go_atomtypes.itp"\n')
        out_file.write(f'#include "go_nbparams.itp"\n')
        out_file.write(f'#include "protein.itp"\n')
        out_file.write(f'#include "solvents.itp"\n')
        out_file.write(f'#include "ions.itp"\n')
        out_file.write(f'\n[ system ]\n')
        out_file.write(f'Martini protein in water\n\n') 
        out_file.write(f'\n[ molecules ]\n')
        out_file.write(f'{protein}  1\n')
    os.chdir(bdir)
        
        
def link_itps(wdir):
    bdir = os.getcwd()
    os.chdir(wdir)
    for name, path in PRT_DICT.items():
        if name.endswith('.itp'):
            command = f'ln -sf {path} {name}' # > /dev/null 2>&
            sp.run(command.split())
    os.chdir(bdir)
    
    
def gmx_pdb(wdir, in_pdb, out_pdb):
    bdir = os.getcwd()
    os.chdir(wdir)
    command = f'gmx_mpi pdb2gmx -f {in_pdb} -o clean.pdb -water none -ff amber94 -renum -ignh' # 
    sp.run(command.split())
    with open(out_pdb, 'w') as fd:
        sp.run(['grep', '^ATOM', 'clean.pdb'], stdout=fd)
    os.remove('clean.pdb')
    os.chdir(bdir)
    
    
def fix_go_map(wdir, in_map, out_map='go.map'):
    bdir = os.getcwd()
    os.chdir(wdir)
    with open (in_map, 'r') as in_file:
         with open (out_map, 'w') as out_file:
            for line in in_file:
                if line.startswith('R '):
                    new_line = ' '.join(line.split()[:-1])
                    out_file.write(f'{new_line}\n')
    os.chdir(bdir)


def prepare_files(pdb, wdir='test', mutations=None, protein='protein'):
    r"""
    -wdir           Relative path to the working directory
    """
    os.makedirs(wdir, exist_ok=True)
    copy_from = os.path.join(wdir, 'protein_minimized.pdb')
    copy_to = os.path.join(wdir, 'protein.pdb')
    shutil.copy(copy_from, copy_to)
    link_itps(wdir)
    make_topology_file(wdir, protein=protein)
    print("Getting Go-map...")
    get_go(wdir, protein)
    fix_go_map(wdir, in_map='protein_map.map')
    print('All the files are ready!')
    
    
def martinize_go(pdb, wdir, go_map='go.map', go_eps=10.0, go_moltype="protein", go_low=0.3, go_up=0.8, go_res_dist=3):
    r"""
    Virtual site based GoMartini:
    -go_map         Contact map to be used for the Martini Go model.Currently, only one format is supported. (default: None)
    -go_moltype     Set the name of the molecule when using Virtual Sites GoMartini. (default: protein)
    -go_eps         The strength of the Go model structural bias in kJ/mol. (default: 9.414)                        
    -go_low         Minimum distance (nm) below which contacts are removed. (default: 0.3)
    -go_up          Maximum distance (nm) above which contacts are removed. (default: 1.1)
    -go_res_dist    Minimum graph distance (similar sequence distance) below whichcontacts are removed. (default: 3)
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    shutil.copy(pdb, 'protein_aa.pdb')
    command = f'martinize2 -f {pdb} -go {go_map} -go-moltype {go_moltype} -go-eps {go_eps} \
        -go-low {go_low} -go-up {go_up} -go-res-dist {go_res_dist} \
        -o protein.top -x protein.pdb -p backbone -dssp -ff martini3001 \
        -sep -scfix -cys 0.3 -resid input -maxwarn 1000'
    sp.run(command.split())
    os.chdir(bdir)
    
    
def solvate(wdir, bt='dodecahedron', d=1.25, radius=0.21, conc=0.0):
    r"""
    -bt             Box type for -box and -d: triclinic, cubic, dodecahedron, octahedron (default: triclinic)
    -d              Distance between the solute and the box (default: 1.25 nm)
    -radius         VWD radius (default: 0.21 nm)
    -conc           Ionic concentration (micromol/l) (default: 0.0 nm)
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    command = f'gmx_mpi editconf -f protein.pdb -c -bt {bt} -d {d} -o system.gro'
    sp.run(command.split())
    command = f'gmx_mpi solvate -cp system.gro -cs {PRT_DICT['water.gro']} -p system.top -radius {radius} -o system.gro'
    sp.run(command.split())
    command = f'gmx_mpi grompp -f {PRT_DICT['ions.mdp']} -c system.gro -p system.top -o ions.tpr -maxwarn 1000'
    sp.run(command.split())
    command = f'gmx_mpi genion -s ions.tpr -p system.top -conc {conc} -neutral -pname NA -nname CL -o system.gro'
    sp.run(command.split(), input='W\n', text=True)
    os.chdir(bdir)
    
    
def energy_minimization(wdir, ncpus=0):
    """
    Perform energy minimization using GROMACS.

    Parameters:
    wdir (str): The working directory where the energy minimization will be performed.
    ncpus (int, optional): Number of CPU threads to use for the minimization. Defaults to 0, 
                           which lets GROMACS decide the number of threads.

    Raises:
    FileNotFoundError: If the necessary input files are not found in the specified directories.
    RuntimeError: If the GROMACS commands fail to execute.
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    os.makedirs('mdrun', exist_ok=True)
    os.chdir('mdrun')
    command = f'gmx_mpi grompp -f {PRT_DICT['em.mdp']} -c ../system.gro -r ../system.gro -p ../system.top -o em.tpr'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm em'
    sp.run(command.split())
    os.chdir(bdir)
    
    
def heatup(wdir, ncpus=0):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('mdrun')
    command = f'gmx_mpi grompp -f {PRT_DICT['hu.mdp']} -c em.gro -r em.gro -p ../system.top -o hu.tpr'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm hu'
    sp.run(command.split())
    os.chdir(bdir)


def equilibration(wdir, ncpus=0):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('mdrun')
    command = f'gmx_mpi grompp -f {PRT_DICT['eq.mdp']} -c hu.gro -r hu.gro -p ../system.top -o eq.tpr'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm eq'
    sp.run(command.split())
    os.chdir(bdir)
    

def md(wdir, nsteps=-2, ncpus=0):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('mdrun')
    command = f'gmx_mpi grompp -f {PRT_DICT['md.mdp']} -c eq.gro -r eq.gro -p ../system.top -o md.tpr'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm md -nsteps {nsteps}'
    sp.run(command.split())
    os.chdir(bdir)
    

def convert_trajectory(wdir, tb=0, te=1000):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.makedirs('analysis', exist_ok=True)
    analysis_dir = os.path.abspath('analysis')
    os.chdir('mdrun')
    shutil.copy('md.tpr', f'{analysis_dir}/md.tpr')
    command = f'gmx_mpi trjconv -s md.tpr -f md.xtc -o {analysis_dir}/mdc.xtc -fit rot+trans -b {tb} -e {te} -tu ns'
    sp.run(command.split(), input='1\n1\n', text=True)
    os.chdir(bdir)
    

def get_covariance_matrix(wdir, tb=100, te=500, tw=5, td=5):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    for t in range(tb, te-tw-td, td):
        b = t
        e = t + tw
        command = f'gmx_mpi trjconv -s ../mdrun/md.tpr -f ../mdrun/md.xtc -o mdc_{b}.xtc -fit rot+trans -b {b} -e {e} -tu ns'
        sp.run(command.split(), input='1\n1\n', text=True)
        command = f'gmx_mpi covar -s md.tpr -f mdc_{b}.xtc -ascii covar_{b}.dat -b {b} -e {e} -tu ns -last 50'
        sp.run(command.split(), input='1\n3\n', text=True)
    files_to_delete = [f for f in os.listdir() if f.startswith('#')]
    for f in files_to_delete:
        os.remove(f)
    os.remove(f'covar_{te}.dat')
    os.chdir(bdir)
    
    
def get_rmsd(wdir, tb=000, te=1000):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    command = f'gmx_mpi rms -s ../mdrun/md.tpr -f mdc.xtc -o rmsd.xvg -b {tb} -e {te} -tu ns -xvg none'
    sp.run(command.split(), input='1\n1\n', text=True)
    files_to_delete = [f for f in os.listdir() if f.startswith('#')]
    for f in files_to_delete:
        os.remove(f)
    os.chdir(bdir)
    
    
def get_rmsf(wdir, tb=000, te=1000):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    tb *= 1000
    te *= 1000
    command = f'gmx_mpi rmsf -s ../mdrun/md.tpr -f mdc.xtc -o rmsf.xvg -b {tb} -e {te} -xvg none -res yes'
    sp.run(command.split(), input='1\n1\n', text=True)
    files_to_delete = [f for f in os.listdir() if f.startswith('#')]
    for f in files_to_delete:
        os.remove(f)
    os.chdir(bdir)    


def parse_covarince_matrix(file):
    print(f"Reading covariance matrix from {file}")
    df = pd.read_csv(file, sep='\\s+', header=None)
    covarince_matrix = np.asarray(df, dtype=np.float64)
    resn = int(np.sqrt(len(covarince_matrix) / 3)) # number of residues
    covarince_matrix = np.reshape(covarince_matrix, (3*resn, 3*resn))
    return covarince_matrix, resn
    
    
def calculate_dfi(covarince_matrix, resn):
    print("Calculating DFI")
    cov = covarince_matrix
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float64)
    norm = np.sqrt(np.sum(directions, axis=1))
    directions = (directions.T / norm).T
    dfi = np.zeros(resn)
    for f in directions:
        f = np.tile(f, resn)
        m = covarince_matrix * f
        m = np.reshape(m, (resn, resn, 3, 3))
        m = np.sum(m, axis=-1)
        pert_mat = np.sqrt(np.sum(m * m, axis=-1))
        dfi += np.sum(pert_mat, axis=-1)
    dfi /= np.sum(dfi)
    return dfi
 
 
def percentile(x):
    """
    Calculate the percentile ranking of each element in a 1-dimensional array.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Array of percentile rankings.
    """
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px 
            
            
def get_dfi(file, ):
    covarince_matrix, resn = parse_covarince_matrix(file)
    dfi = calculate_dfi(covarince_matrix, resn)
    pdfi = percentile(dfi)
    print("Saving DFI")
    data = pd.DataFrame()
    resnums = list(range(26, resn + 26))
    data["resn"] = resnums
    data["dfi"] = dfi
    data["pdfi"] = pdfi
    data.to_csv(file.replace('covar', 'dfi'), index=False, header=None, sep=' ')
    return resnums, dfi


def get_dfis(wdir):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    covar_files = [f for f in os.listdir() if f.startswith('covar') and f.endswith('.dat')]
    dfis = []
    for file in covar_files:
        resnums, dfi = get_dfi(file)
        dfis.append(dfi)
    dfi_av = np.average(np.array(dfis).T, axis=-1)
    data = pd.DataFrame()
    data["resn"] = resnums
    data["dfi"] = dfi_av
    data["pdfi"] = percentile(dfi_av)
    data.to_csv('dfi_av.dat', index=False, header=None, sep=' ')
    # for file in covar_files:
    #     os.remove(file)
    os.chdir(bdir)
    
    
def parse_dfi(file):
    df = pd.read_csv(file, sep=' ', header=None)
    res = df[0]
    dfi = df[1]
    pdfi = df[2]
    return res, dfi, pdfi
    
    
def plot_dfi(wdir):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    dfi_files = [f for f in os.listdir() if f.startswith('dfi_av') and f.endswith('.dat')]
    fig = plt.figure(figsize=(12,4))
    for file in dfi_files:
        res, dfi, pdfi = parse_dfi(file)
        plt.plot(res, pdfi)
    plt.autoscale(tight=True)
    plt.grid()
    plt.tight_layout()
    fig.savefig(f'dfi.png')
    plt.close()
    os.chdir(bdir)    
    

if __name__  == "__main__":
    pass



