import os
import sys
import importlib.resources
import MDAnalysis as mda
import numpy as np
import pandas as pd
import shutil
import subprocess as sp
# reforge
from reforge import cli, mdm, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger
# mm
import openmm as mm
from openmm import app
from openmm.unit import *


################################################################################
# CG system class
################################################################################   

class mmSystem:
    """
    Class to set up and analyze protein-nucliotide-lipid systems for MD with GROMACS
    All the attributes are the paths to files and directories needed to set up and run CG MD
    """    
    MDATDIR = importlib.resources.files("reforge") / "martini" / "data" 
    MITPDIR = importlib.resources.files("reforge") / "martini" / "itp" 
    NUC_RESNAMES = ['A', 'C', 'G', 'U', 'RA3', 'RA5', 'RC3', 'RC5', 'RG3', 'RG5', 'RU3', 'RU5']
    
    def __init__(self, sysdir, sysname, **kwargs):
        """
        Initializes the system with the required directories and files.
        
        Args:
            sysdir (str): Directory for the system files.
            sysname (str): Name of the system.
            pdb (str): PDB file of the system.
            kwargs: Additional keyword arguments.
            
        Sets up paths to various files required for CG MD simulation.
        """
        self.sysname    = sysname
        self.sysdir     = os.path.abspath(sysdir)
        self.wdir       = os.path.join(self.sysdir, sysname)
        self.inpdb      = os.path.join(self.wdir, 'inpdb.pdb')
        self.syspdb     = os.path.join(self.wdir, 'system.pdb')
        self.sysxml     = os.path.join(self.wdir, 'system.xml')
        self.mdcpdb     = os.path.join(self.wdir, 'mdc.pdb')
        self.trjpdb     = os.path.join(self.wdir, 'traj.pdb')
        self.prodir     = os.path.join(self.wdir, 'proteins')
        self.nucdir     = os.path.join(self.wdir, 'nucleotides')
        self.iondir     = os.path.join(self.wdir, 'ions')
        self.ionpdb     = os.path.join(self.iondir, 'ions.pdb')
        self.topdir     = os.path.join(self.wdir, 'topol')
        self.mapdir     = os.path.join(self.wdir, 'map')
        self.mdpdir     = os.path.join(self.wdir, 'mdp')
        self.cgdir      = os.path.join(self.wdir, 'cgpdb')
        self.mddir      = os.path.join(self.wdir, 'mdruns')
        self.datdir     = os.path.join(self.wdir, 'data')
        self.pngdir     = os.path.join(self.wdir, 'png')
        self._chains    = []
        self._mdruns    = []

    @property
    def chains(self):
        """
        A list of chain ids in the system. 
        Either provide or extract from the PDB file
        Returns:
            list: List of chain identifiers.
        """
        if self._chains:
            return self._chains
        chain_names = set()
        with open(self.syspdb, 'r') as file:
            for line in file:
                # Look for lines that define atomic coordinates
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21].strip()  # Chain ID is in column 22 (index 21)
                    if chain_id:  # Only add non-empty chain IDs
                        chain_names.add(chain_id)
        self._chains = sort_uld(chain_names)
        return self._chains
    
    @chains.setter
    def chains(self, chains):
        self._chains = chains
        
    @property
    def mdruns(self):
        """
        A list of mdruns. Either provide or look up in self.mddir
        Returns:
            list: List of chain identifiers.
        """
        if self._mdruns:
            return self._mdruns
        if not os.path.isdir(self.mddir):
            return self._mdruns
        for adir in sorted(os.listdir(self.mddir)):
            dir_path = os.path.join(self.mddir, adir)
            if os.path.isdir(dir_path):
                self._mdruns.append(adir)
        return self._mdruns
    
    @mdruns.setter
    def mdruns(self, mdruns):
        self._mdruns = mdruns

    def prepare_files(self):
        """
        Creates the necessary directories for the simulation and copies the 
        relevant input files to the working directory.
        """
        print('Preparing files and directories', file=sys.stderr)
        os.makedirs(self.prodir, exist_ok=True)
        os.makedirs(self.nucdir, exist_ok=True)
        os.makedirs(self.topdir, exist_ok=True)
        os.makedirs(self.mapdir, exist_ok=True)
        os.makedirs(self.mdpdir, exist_ok=True)
        os.makedirs(self.cgdir,  exist_ok=True)
        os.makedirs(self.grodir, exist_ok=True)
        os.makedirs(self.datdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        for file in os.listdir(self.MDATDIR):
            if file.endswith('.mdp'):
                fpath = os.path.join(self.MDATDIR, file)
                outpath = os.path.join(self.mdpdir, file)
                shutil.copy(fpath, outpath)
        shutil.copy(os.path.join(self.MDATDIR, 'water.gro'), self.wdir)
        for file in os.listdir(self.MITPDIR):
            if file.endswith('.itp'):
                fpath = os.path.join(self.MITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                shutil.copy(fpath, outpath)

    def clean_pdb(self, pdb_file, **kwargs):
        """
        Cleans starting PDB file using PDBfixer by OpenMM
        """
        print("Cleaning the PDB", file=sys.stderr)
        pdbtools.clean_pdb(pdb_file, self.inpdb, **kwargs)

    @staticmethod
    def forcefield(force_field='amber14-all.xml', water_model='amber14/tip3p.xml', **kwargs):
        forcefield = app.ForceField(force_field, water_model, **kwargs)
        return forcefield   

    @staticmethod
    def modeller(inpdb, forcefield, **kwargs):
        kwargs.setdefault('model', 'tip3p') 
        kwargs.setdefault('boxShape', 'dodecahedron') 
        kwargs.setdefault('padding', 1.0*nanometer) 
        kwargs.setdefault('positiveIon', 'Na+') 
        kwargs.setdefault('negativeIon', 'Cl-') 
        kwargs.setdefault('ionicStrength', 0.1*molar)    
        pdb = app.PDBFile(inpdb) 
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(forcefield, **kwargs)
        return modeller        

    def model(self, forcefield, modeller, barostat=None, thermostat=None, **kwargs):
        kwargs.setdefault('nonbondedMethod', app.PME) 
        kwargs.setdefault('nonbondedCutoff', 1.0*nanometer) 
        kwargs.setdefault('constraints', app.HBonds) 
        model = forcefield.createSystem(modeller.topology, **kwargs)
        barostat and model.addForce(barostat) # add barostat if given
        thermostat and model.addForce(thermostat)
        # Saving files
        with open(self.syspdb, 'w') as file:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, file, keepIds=True)
        with open(self.sysxml, 'w') as file:
            file.write(mm.XmlSerializer.serialize(model))
        return model


################################################################################
# MDRun class
################################################################################   

class mdRun(mmSystem):
    """
    Run molecular dynamics (MD) simulation using the specified input files.
    This method runs the MD simulation by calling an external simulation software, 
    such as GROMACS, with the provided input files.
    """
    def __init__(self, sysdir, sysname, runname):
        """
        Initializes required directories and files.
        
        Args:
            runname (str): 
            rundir (str): Directory for the run files.
            rmsdir (str): rms analysis directory
            covdir (str): covariance analysis dir
            lrtdir (str): covariance analysis dir
            cludir (str): clustering dir
            pngdir (str): figures
            kwargs: Additional keyword arguments.
        """
        super().__init__(sysdir, sysname)
        self.runname = runname
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, 'rms_analysis')
        self.covdir = os.path.join(self.rundir, 'cov_analysis')
        self.lrtdir  = os.path.join(self.rundir, 'lrt_analysis')
        self.cludir = os.path.join(self.rundir, 'clusters')
        self.pngdir = os.path.join(self.rundir, 'png')
        
    def prepare_files(self):
        """
        Create necessary directories.
        """
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)

    def modeller(self):
        pdb = app.PDBFile(self.syspdb) 
        modeller = app.Modeller(pdb.topology, pdb.positions)
        return modeller

    def simulation(self, modeller, integrator):
        simulation = app.Simulation(modeller.topology, self.sysxml, integrator)
        simulation.context.setPositions(modeller.positions)
        return simulation

    def save_state(self, simulation, file_prefix='sim'):
        pdb_file = os.path.join(self.rundir, file_prefix + '.pdb')
        xml_file = os.path.join(self.rundir, file_prefix + '.xml')
        simulation.saveState(xml_file)
        state = simulation.context.getState(getPositions=True)   
        positions = state.getPositions()
        with open(pdb_file, 'w') as file:
            app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)
        
    def em(self, simulation, tolerance=100, maxIterations=1000):
        print('Minimizing energy...', file=sys.stderr)
        # Files
        log_file = os.path.join(self.rundir, 'em.log')
        # Reporters
        reporter = app.StateDataReporter(log_file, 100, step=True, potentialEnergy=True, temperature=True)
        # Simulation
        simulation.reporters.append(reporter)
        simulation.minimizeEnergy(tolerance, maxIterations)
        self.save_state(simulation, 'em')
        print('Minimization complete.', file=sys.stderr)

    def eq(self, simulation, nsteps=10000, nlog=10000, **kwargs):
        print('Starting equilibration...')
        kwargs.setdefault('step', True) 
        kwargs.setdefault('potentialEnergy', True) 
        kwargs.setdefault('temperature', True) 
        kwargs.setdefault('density', True)
        # Files
        em_xml = os.path.join(self.rundir, 'em.xml')
        log_file = os.path.join(self.rundir, 'eq.log')
        # Reporters
        reporter = app.StateDataReporter(log_file, nlog, **kwargs)
        # Simulation
        simulation.loadState(em_xml)
        simulation.reporters.append(reporter)
        simulation.step(nsteps)
        self.save_state(simulation, 'eq')
        print('Equilibration complete.')

    def md(self, simulation, nsteps=100000, nout=1000, nlog=10000, nchk=10000, **kwargs):
        print('Production run...')
        kwargs.setdefault('step', True) 
        kwargs.setdefault('time', True) 
        kwargs.setdefault('potentialEnergy', True) 
        kwargs.setdefault('temperature', True) 
        kwargs.setdefault('density', False) 
        # Files
        eq_xml = os.path.join(self.rundir, 'eq.xml')
        trj_file = os.path.join(self.rundir, 'md.dcd')
        log_file = os.path.join(self.rundir, 'md.log')
        xml_file = os.path.join(self.rundir, 'md.xml')
        pdb_file = os.path.join(self.rundir, 'md.pdb')
        # Reporters
        trj_reporter = app.DCDReporter(trj_file, nout, append=False)
        pdb_reporter = app.PDBReporter(pdb_file, nchk)
        log_reporter = app.StateDataReporter(log_file, nlog, **kwargs)
        xml_reporter = app.CheckpointReporter(xml_file, nchk, writeState=True)
        reporters = [trj_reporter, log_reporter, xml_reporter]
        # Simulation
        simulation.loadState(eq_xml)
        simulation.reporters.extend(reporters)
        simulation.step(nsteps)
        self.save_state(simulation, 'md')
        print('Production complete.')


################################################################################
# Helper functions
################################################################################   

def sort_uld(alist):
    """
    Sorts characters in a list such that they appear in the following order: 
    uppercase letters first, then lowercase letters, followed by digits. 
    Helps with orgazing gromacs multichain files
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist

