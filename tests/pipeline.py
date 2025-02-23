import os
import numpy as np
import pandas as pd
import sys
import shutil
import cgtools.cli as cli
import cgtools.lrt as lrt
import MDAnalysis as mda
from cgtools.gmxmd import gmxSystem, MDRun
from cgtools.dataio import read_data, pull_all_files, filter_files
from cgtools.utils import *
from pathlib import Path


def setup(sysdir, sysname):

    ### FOR COARSE-GRAINED MODELS ###
    system = gmxSystem(sysdir, sysname)

    # 1.1. Need to copy force field and md-parameter files and prepare directories
    system.prepare_files()

    # 1.2. Try to clean the input PDB and split the chains based on the type of molecules (protein, RNA/DNA)
    system.sort_input_pdb("some_stuff.pdb")
    system.clean_inpdb(add_missing_atoms=False, add_hydrogens=True, pH=7.0)
    system.split_chains()
    # system.clean_chains(add_missing_atoms=True, add_hydrogens=True, pH=7.0)  # if didn't work for the whole PDB

    # 1.3. COARSE-GRAINING. Done separately for each chain. If don't want to split some of them, it needs to be done manually.
    # system.get_go_maps()
    # system.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p='backbone', pf=500) # Martini + Go-network FF
    system.martinize_proteins_en(ef=1000, el=0.3, eu=0.8, p='backbone', pf=500, append=False)  # Martini + Elastic network FF 
    system.martinize_rna(ef=200, el=0.3, eu=1.2, p='backbone', pf=500, append=False) # Martini RNA FF 
    system.make_martini_topology_file(add_resolved_ions=False, prefix='chain') # CG topology
    system.make_cgpdb_file(bt='dodecahedron', d='1.2', ) # CG structure

    # 1.4. Coarse graining is done. s and then add solvent and ions
    system.solvate()
    system.add_bulk_ions(conc=0.15, pname='NA', nname='CL')

    # 1.5. Need index files to make selection with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. 
    # Can add custom groups using AtomList.write_to_ndx method
    system.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])
   
      
def md(sysdir, sysname, runname, ntomp): 
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    mdrun.empp()
    mdrun.mdrun(deffnm='em', ntomp=ntomp)
    mdrun.eqpp(c='em.gro', r='em.gro', maxwarn=10) 
    mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    mdrun.mdpp(c='eq.gro', r='eq.gro')
    mdrun.mdrun(deffnm='md', ntomp=ntomp) 
    
    
def extend(sysdir, sysname, runname, ntomp):    
    system = gmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', ntomp=ntomp, nsteps=-2) 


def make_ndx(sysdir, sysname, **kwargs):
    system = gmxSystem(sysdir, sysname)
    system.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])
      
    
def trjconv(sysdir, sysname, runname, mode='solu', fit='rot+trans', **kwargs):
    kwargs.setdefault('b', 0) # in ps
    kwargs.setdefault('dt', 1000) # in ps
    kwargs.setdefault('e', 100000) # in ps
    mdrun = MDRun(sysdir, sysname, runname)
    if mode == 'solu': # REMOVE SOLVENT # NDX groups: 1.System 2.Solute 3.Backbone 4.Solvent 5...chains...
        k = 1
    if mode == 'bb': # FOR BACKBONE ANALYSIS
        k = 2
    mdrun.trjconv(clinput='k\nk\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='whole', ur='compact', e=0)
    mdrun.trjconv(clinput='k\nk\n', s='md.tpr', f='md.trr', o='mdc.trr', n=mdrun.sysndx, pbc='whole', ur='compact', **kwargs)
    mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.trr', o='mdc.trr', pbc='nojump')
    if fit:
        mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.trr', o='mdc.trr', fit=fit)
    clean_dir(mdrun.rundir)
    

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    system = gmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 5000
    e = 10000000
    f = 'traj.trr'
    s = 'traj.pdb'
    # mdrun.rmsf(clinput=f'0\n 0\n', f=f, s=s, b=b, e=e, n=system.trjndx, res='no', fit='yes',  )

    
def cluster(sysdir, sysname, runname, **kwargs):
    system = gmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 400000
    mdrun.cluster(b=b, dt=1000, cutoff=0.15, method='gromos', cl='clusters.pdb', clndx='cluster.ndx', av='yes')
    mdrun.extract_cluster()
    
    
def lrt(sysdir, sysname, runname):
    system = gmxSystem(sysdir, sysname)
    run = system.initmd(runname)
    run.prepare_files()
    run.get_covmats(f='../traj.trr', s='../traj.pdb', b=0, n=1)
    run.get_pertmats()
    run.get_dfi()
    run.get_full_dci(asym=False)
    # # group dci
    # u = mda.Universe(self.trjpdb)
    # groups = u.segments.ids
    # group_ids = [s.segid for s in u.segments]
    # run.get_group_dci(asym=False)



def tdlrt(sysdir, sysname, runname):
    system = gmxSystem(sysdir, sysname)
    run = system.initmd(runname)
    run.prepare_files()
    bdir = os.getcwd()
    os.chdir(run.covdir)
    # CCF params FRAMEDT=20 ps
    nskip = 1
    ntmax = 1000
    tag = 'pv'   
    corr_file = f'corr_{tag}.npy'
    # # CALC CCF
    # print(f'Working directory: {os.getcwd()}', file=sys.stderr) 
    # pos, vel = lrt.read_trajectory(resp_ids=[], pert_ids=[], f='../traj.trr', s='../traj.pdb', b=200000, e=1000000, skip_rate=nskip, dtype=np.float32)
    # print(f'Read {pos.shape} positions and velocities', file=sys.stderr)
    # tag2args = {'pp': (pos, pos), 'pv': (pos, vel), 'vv': (vel, vel), }
    # corr = lrt.calc_ccf(*tag2args[tag], ntmax=ntmax, n=10, mode='gpu', center=True)
    # np.save(corr_file, corr)
    # VIDEO
    make_animation(infile=corr_file, nframes=ntmax, outfile=f'{bdir}/data/{tag}_{sysname}_{runname}.mp4')
    os.chdir(bdir)


def get_averages(sysdir, sysname, rmsf=True, dfi=True, dci=True, ):
    system = gmxSystem(sysdir, sysname)  
    print('Starting', file=sys.stderr )    
    all_files = pull_all_files(system.mddir)
    if rmsf:  # RMSF
        print(f'Processing RMSF', file=sys.stderr )
        files = filter_files(all_files, sw='rmsf.', ew='.xvg')
        system.get_mean_sem(files, f'rmsf.csv', col=1)
        # Chain RMSF
        for chain in system.chains:
            print(f'Processing chain {chain}', file=sys.stderr )
            sw = f'rmsf_{chain}'
            files = filter_files(all_files, sw=sw, ew='.xvg')
            system.get_mean_sem(files, f'{sw}.csv', col=1)
    if dfi:  # DFI
        print(f'Processing DFI', file=sys.stderr )
        files = filter_files(all_files, sw='dfi', ew='.xvg')
        system.get_mean_sem(files, f'dfi.csv', col=1)
    if dci: # DCI
        print(f'Processing DCI', file=sys.stderr )
        files = filter_files(all_files, sw='dci', ew='.xvg')
        system.get_mean_sem_2d(files, out_fname=f'dci.csv', out_errname=f'dci_err.csv')
        
    
def plot_averages(sysdir, sysname, **kwargs):    
    from plotting import plot_each_mean_sem
    system = gmxSystem(sysdir, sysname)  
    for metric in ['rmsf', ]:
        fpaths = [os.path.join(system.datdir, f) for f in os.listdir(system.datdir) if f.startswith(metric)]
        plot_each_mean_sem(fpaths, system)


def test(sysdir, sysname, runname, **kwargs):    
    print('passed', file=sys.stderr)

        
if __name__ == '__main__':
    todo = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        "setup": setup,
        "md": md,
        "extend": extend,
        "make_ndx": make_ndx,
        "trjconv": trjconv,
        "rms_analysis": rms_analysis,
        "cluster": cluster,
        "dci_dfi": dci_dfi,
        "tdlrt": tdlrt,
        "get_averages": get_averages,
        "plot": plot_averages,
        "test": test,
    }
    if todo in commands:   # Then, assuming `todo` is the command name (a string)
        commands[todo](*args) # `args` is a list/tuple of arguments
    else:
        raise ValueError(f"Unknown command: {todo}")
   
        
    