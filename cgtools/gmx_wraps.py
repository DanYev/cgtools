import os
import subprocess as sp
import shutil


def mdrun(mddir, mdp='../mdp/em.mdp', c='../system.gro', r='../system.gro', p='../system.top', 
    o='em.tpr', deffnm='em', ncpus=0):
    """
    Prepare files for mdrun and run it

    Parameters:
    mddir (str): The working directory where the energy minimization will be performed.
    ncpus (int, optional): Number of CPU threads to use for the minimization. Defaults to 0, 
                           which lets GROMACS decide the number of threads.

    Raises:
    FileNotFoundError: If the necessary input files are not found in the specified directories.
    RuntimeError: If the GROMACS commands fail to execute.
    """
    bdir = os.getcwd()
    os.chdir(mddir)
    os.makedirs(mddir, exist_ok=True)
    os.chdir(mddir)
    command = f'gmx_mpi grompp -f {mdp} -c {c} -r {r} -p {p} -o {o}'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm {deffnm}'
    sp.run(command.split())
    os.chdir(bdir)
    

def energy_minimization(wdir, mddir, mdp='../mdp/em.mdp', deffnm='em', ncpus=0):
    """
    Perform energy minimization using GROMACS.

    Parameters:
    mddir (str): The working directory where the energy minimization will be performed.
    ncpus (int, optional): Number of CPU threads to use for the minimization. Defaults to 0, 
                           which lets GROMACS decide the number of threads.

    Raises:
    FileNotFoundError: If the necessary input files are not found in the specified directories.
    RuntimeError: If the GROMACS commands fail to execute.
    """
    bdir = os.getcwd()
    os.chdir(mddir)
    os.makedirs(mddir, exist_ok=True)
    os.chdir(mddir)
    command = f'gmx_mpi grompp -f {mdp} -c ../system.gro -r ../system.gro -p ../system.top -o em.tpr'
    sp.run(command.split())
    options = f'-ntomp {ncpus} -pin on -pinstride 1'
    command = f'gmx_mpi mdrun {options} -deffnm {deffnm}'
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