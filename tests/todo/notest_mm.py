import numpy as np
import openmm as mm
import os
import pandas as pd
import sys
import shutil
import MDAnalysis as mda
from reforge import cli, lrt
from reforge.mmmd import mmSystem, mdRun
from openmm import app
from openmm.unit import *
from pathlib import Path


def setup(sysdir, sysname, mode='cg'):
    system = mmSystem(sysdir, sysname)
    inpdb = os.path.join(system.wdir, '1btl.pdb')
    system.clean_pdb(inpdb, add_missing_atoms=True, add_hydrogens=True)
    forcefield = system.forcefield()
    modeller = system.modeller(system.inpdb, forcefield)
    barostat = mm.MonteCarloBarostat(1*bar, 300*kelvin)
    model = system.model(forcefield, modeller, barostat=barostat)
      
    
def md(sysdir, sysname, runname, ntomp): 
    mdrun = mdRun(sysdir, sysname, 'mdrun')
    mdrun.prepare_files()
    # Production parameters
    tstep = 2*femtoseconds
    tot_time = 0.2*nanoseconds 
    nsteps = int(tot_time / tstep)
    temperature = 300*kelvin
    # Prep
    modeller = mdrun.modeller() # Load the model when called by mdrun
    integrator = mm.LangevinMiddleIntegrator(temperature, 1/picosecond, 1*femtoseconds)
    simulation = mdrun.simulation(modeller, integrator)
    # EM 
    mdrun.em(simulation, tolerance=1, maxIterations=1000)
    # EQ
    mdrun.eq(simulation, nsteps=100000)
    # MD run
    simulation.integrator.setStepSize(tstep)
    mdrun.md(simulation, nsteps=nsteps)

    
def extend(sysdir, sysname, runname, ntomp):    
    system = gmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.mdrun(deffnm='md', cpi='md.cpt', ntomp=ntomp, nsteps=-2) 
    
    
def trjconv(sysdir, sysname, runname, mode='aa', ):
    system = gmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    shutil.copy('atommass.dat', os.path.join(mdrun.rundir, 'atommass.dat'))
    if mode == 'solu': # FOR PROTEIN ANALYSIS
        mdrun.trjconv(clinput='1\n1\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='whole', ur='compact', e=0)
        mdrun.trjconv(clinput='1\n1\n', s='md.tpr', f='md.trr', o='mdc.xtc', n=mdrun.sysndx, pbc='whole', ur='compact', dt=1000, e=1000000)
        mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.xtc', o='mdc.xtc', pbc='nojump')
        mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.xtc', o='traj.xtc', fit='rot+trans')
    if mode == 'cg': # FOR BACKBONE ANALYSIS
        mdrun.trjconv(clinput='1\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', e=0)
        mdrun.trjconv(clinput='1\n', s='md.tpr', f='md.trr', o='mdc.trr', n=mdrun.sysndx, pbc='atom', ur='compact', e=1000000)
        mdrun.trjconv(clinput='0\n', f='mdc.trr', o='mdc.trr', pbc='nojump')
        mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.pdb', o='traj.pdb', n=mdrun.bbndx, fit='rot+trans', e=0)
        mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.trr', o='traj.trr', n=mdrun.bbndx, fit='rot+trans', b=0,)
    if mode == 'aa': # FOR ALL-ATOM BB
        mdrun.trjconv(clinput='3\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', e=0)
        mdrun.trjconv(clinput='3\n', s='md.tpr', f='md.trr', o='mdc.trr', n=mdrun.sysndx, pbc='atom', ur='compact', e=1000000)
        mdrun.trjconv(clinput='0\n', f='mdc.trr', o='mdc.trr', pbc='nojump')
        mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.pdb', o='traj.pdb', fit='rot+trans', e=0)
        mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.trr', o='traj.trr', fit='rot+trans', b=0,)


def test(sysdir, sysname, runname, **kwargs):    
    print('passed', file=sys.stderr)

        
if __name__ == '__main__':
    todo = sys.argv[1]
    args = sys.argv[2:]
    match todo:
        case 'setup':
            setup(*args)
        case 'md':
            md(*args)    
        case 'extend':
            extend(*args)
        case 'trjconv':
            trjconv(*args,)
        case 'test':
            test(*args)
   
        
    