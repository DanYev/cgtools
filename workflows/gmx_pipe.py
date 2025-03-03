"""
Gmx Pipe Tutorial
=================

This module serves as a tutorial for setting up and running coarse-grained
molecular dynamics (MD) simulations using the reForge package and GROMACS. It
provides a pipeline that performs several key tasks:
    - System Setup: Copying force field and parameter files, preparing directories,
    cleaning and sorting the input PDB file, and splitting chains.
    - Coarse-Graining: Applying coarse-graining methods for proteins (using both
    Go-model and elastic network approaches) and nucleotides.
    - Solvation and Ion Addition: Solvating the system and adding ions to neutralize it.
    - MD Simulation: Running energy minimization, equilibration, and production MD runs.
    - Post-Processing: Converting trajectories, and performing
    analyses such as RMSF, RMSD, covariance analysis, clustering, and time-dependent
    correlation calculations.

Author: DY
Date: 2025-XX-XX
"""

import os
import numpy as np
import pandas as pd
import sys
import shutil
import MDAnalysis as mda
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *
from pathlib import Path


def setup(*args):
    # setup_cg_protein_rna(*args)
    setup_cg_protein_membrane(*args)


def setup_cg_protein_rna(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    mdsys.prepare_files() # be careful it can overwrite later files
    mdsys.sort_input_pdb(f"{sysname}.pdb") # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # # 1.2.1 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    # mdsys.clean_pdb_mm(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    # mdsys.split_chains()
    # mdsys.clean_chains_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB
    
    # 1.2.2 Same but if we want Go-Model for the proteins
    mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, sometimes you need to refer to AMBER FF
    mdsys.split_chains()
    mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    mdsys.get_go_maps(append=True)

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=700, el=0.3, eu=0.8, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p='backbone', pf=500, append=False) # Martini + Go-network FF
    mdsys.martinize_rna(ef=200, el=0.3, eu=1.2, p='backbone', pf=500, append=True) # Martini RNA FF 
    mdsys.make_cg_topology(add_resolved_ions=False, prefix='chain') # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure(bt='dodecahedron', d='1.2', ) # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    
    # 1.4. Coarse graining is *hopefully* done. Need to add solvent and ions
    solvent = os.path.join(mdsys.wdir, 'water.gro')
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent) # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.15, pname='NA', nname='CL')

    # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])
   
      
def setup_cg_protein_membrane(sysdir, sysname):
    ### FOR CG PROTEIN+LIPID BILAYERS ###
    mdsys = GmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    # mdsys.prepare_files() # be careful it can overwrite later files
    # mdsys.sort_input_pdb(f"{sysname}.pdb") # sorts chain and atoms in the input file and returns makes mdsys.inpdb file

    # # 1.2.1 Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    # mdsys.clean_pdb_mm(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    # mdsys.split_chains()
    # mdsys.clean_chains_mm(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB
    
    # 1.2.2 Same but if we want Go-Model for the proteins
    # mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput='8\n 7\n', ignh='no', renum='yes') # 8 for CHARMM, sometimes you need to refer to AMBER FF
    # mdsys.split_chains()
    # mdsys.clean_chains_gmx(clinput='8\n 7\n', ignh='yes', renum='yes')
    # mdsys.get_go_maps(append=True)

    # # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually. 
    # mdsys.martinize_proteins_en(ef=500, el=0.3, eu=0.8, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    # mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p='backbone', pf=500, append=True) # Martini + Go-network FF
    # mdsys.make_cg_topology(add_resolved_ions=False, prefix='chain') # CG topology. Returns mdsys.systop ("mdsys.top") file
    # mdsys.make_cg_structure(bt='dodecahedron', d='1.2', ) # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.insert_membrane(
        f=mdsys.solupdb, o=mdsys.syspdb, p=mdsys.systop, 
        x=20, y=20, z=20, 
        l='POPC:1', u='POPC:1', sol='W',
    )
    exit()


    # 1.4. Coarse graining is hopefully done. Need to add solvent and ions
    solvent = os.path.join(mdsys.wdir, 'water.gro')
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent) # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.15, pname='NA', nname='CL')

    # 1.5. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


def md(sysdir, sysname, runname, ntomp): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    # Choose appropriate mdp files
    em_mdp = os.path.join(mdrun.mdpdir, 'em.mdp')
    eq_mdp = os.path.join(mdrun.mdpdir, 'eq.mdp')
    md_mdp = os.path.join(mdrun.mdpdir, 'md.mdp')
    mdrun.empp(f=em_mdp) # Preprocessing 
    mdrun.mdrun(deffnm='em', ntomp=ntomp) # Actual run
    mdrun.eqpp(f=eq_mdp, c='em.gro', r='em.gro', maxwarn=10) 
    mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    mdrun.mdpp(f=md_mdp, c='eq.gro', r='eq.gro')
    mdrun.mdrun(deffnm='md', ntomp=ntomp) 
    
    
def extend(sysdir, sysname, runname, ntomp):    
    mdsys = GmxSystem(sysdir, sysname)
    mdrun = mdsys.initmd(runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', ntomp=ntomp, nsteps=-2) 


def make_ndx(sysdir, sysname, **kwargs):
    mdsys = GmxSystem(sysdir, sysname)
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])
      
    
def trjconv(sysdir, sysname, runname, mode='solu', fit='rot+trans', **kwargs):
    kwargs.setdefault('b', 0) # in ps
    kwargs.setdefault('dt', 1000) # in ps
    kwargs.setdefault('e', 1000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    if mode == 'solu': # REMOVE SOLVENT # NDX groups: 1.System 2.Solute 3.Backbone 4.Solvent 5...chains...
        k = 1
    if mode == 'bb': # JUST FOR BACKBONE ANALYSIS
        k = 2
    # mdrun.trjconv(clinput=f'{k}\n {k}\n {k}\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, center='yes', pbc='cluster', ur='compact', e=0)
    # mdrun.trjconv(clinput=f'{k}\n {k}\n {k}\n', s='md.tpr', f='md.trr', o='mdc.trr', n=mdrun.sysndx, center='yes', pbc='cluster', ur='compact', **kwargs)
    # mdrun.trjconv(clinput='0\n 0\n', s='mdc.pdb', f='mdc.trr', o='mdc.trr', pbc='nojump')
    if fit:
        # mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.trr', o='mdc.trr', fit=fit)
        mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.trr', o='mdv.pdb', dt=10000)
    clean_dir(mdrun.rundir)
    

def rms_analysis(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault('b',  50000) # in ps
    kwargs.setdefault('dt', 1000) # in ps
    kwargs.setdefault('e', 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.rmsf(clinput=f'2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='yes', res='yes', **kwargs) # n=mdrun.sysndx
    mdrun.rmsd(clinput=f'2\n 2\n', s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit='rot+trans', **kwargs)

    
def cluster(sysdir, sysname, runname, **kwargs):
    mdrun = GmxRun(sysdir, sysname, runname)
    b = 100000
    mdrun.cluster(clinput=f'1\n 1\n', b=b, dt=1000, cutoff=0.15, method='gromos', cl='clusters.pdb', clndx='cluster.ndx', av='yes')
    mdrun.extract_cluster()


def cov_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname) 
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms.select_atoms("name CA or name P or name C1'") # Select the backbone atoms
    if not ag:
        ag = u.atoms.select_atoms("name BB or name BB1 or name BB3")
    # clean_dir(mdrun.covdir, '*npy')
    mdrun.get_covmats(u, ag, sample_rate=1, b=50000, e=1000000, n=4, outtag='covmat') #  Begin at b picoseconds, end at e, sample each frame
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag='dfi')
    mdrun.get_dci(outtag='dci', asym=False)
    mdrun.get_dci(outtag='asym', asym=True)
    # clean_dir(mdrun.covdir, 'covmat*')


def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname) 
    # CCF params FRAMEDT=20 ps
    b = 0
    e = 100000
    sample_rate = 1
    ntmax = 1000 # how many frames to save
    tag = 'pv'
    corr_file = os.path.join(mdrun.lrtdir, f'corr_{tag}.npy')
    # CALC CCF
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms
    positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e) 
    velocities = io.read_velocities(u, ag, sample_rate=sample_rate, b=b, e=e)
    corr = mdm.ccf(positions, velocities, ntmax=ntmax, n=5, mode='gpu', center=True)
    np.save(corr_file, corr)


def get_averages(sysdir, sysname):
    mdsys = GmxSystem(sysdir, sysname)   
    mdsys.get_mean_sem(pattern='pertmat*.npy')
    mdsys.get_mean_sem(pattern='covmat*.npy')
    # mdsys.get_mean_sem(pattern='dfi*.npy')
    # mdsys.get_mean_sem(pattern='dci*.npy')
    # mdsys.get_mean_sem(pattern='asym*.npy')
    # mdsys.get_mean_sem(pattern='rmsf*.npy')


def get_td_averages(sysdir, sysname, loop=True, fname='corr_pv.npy'):
    """
    Need to loop for big arrays
    """
    mdsys = GmxSystem(sysdir, sysname)  
    print('Getting averages', file=sys.stderr)  
    files = io.pull_files(mdsys.mddir, fname)
    if loop:
        print(f'Processing {files[0]}', file=sys.stderr) 
        average = np.load(files[0])
        for f in files[1:]:
            print(f'Processing {f}', file=sys.stderr)  
            arr = np.load(f)
            average += arr
        average /= len(files)
    else:
        arrays = [np.load(f) for f in files]
        average = np.average(arrays, axis=0)
    np.save(os.path.join(mdsys.datdir, fname), average) 
    print('Done!', file=sys.stderr)  
    return average


def test(sysdir, sysname, runname, **kwargs):    
    print('passed', file=sys.stderr)

        
if __name__ == '__main__':
    command = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        "setup": setup,
        "md": md,
        "extend": extend,
        "make_ndx": make_ndx,
        "trjconv": trjconv,
        "rms_analysis": rms_analysis,
        "cluster": cluster,
        "cov_analysis": cov_analysis,
        "tdlrt_analysis": tdlrt_analysis,
        "get_averages": get_averages,
        "get_td_averages": get_td_averages,
        "test": test,
    }
    if command in commands:   # Then, assuming `command` is the command name (a string)
        commands[command](*args) # `args` is a list/tuple of arguments
    else:
        raise ValueError(f"Unknown command: {command}")
   
        
    