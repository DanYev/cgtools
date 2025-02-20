import os
import sys
import importlib.resources
import MDAnalysis as mda
import numpy as np
import pandas as pd
import shutil
import subprocess as sp
# cgtools
from cgtools import cli, pdbtools, lrt
# mm
import openmm as mm
from openmm import app
from simtk.unit import *

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


################################################################################
# CG system class
################################################################################   

class mmSystem:
    """
    Class to set up and analyze protein-nucliotide-lipid systems for MD with GROMACS
    All the attributes are the paths to files and directories needed to set up and run CG MD
    """    
    MDATDIR = importlib.resources.files("cgtools") / "martini" / "data" 
    MITPDIR = importlib.resources.files("cgtools") / "martini" / "itp" 
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

    def md_system(self, **kwargs):
        inpdb = kwargs.pop('inpdb', self.inpdb)
        force_field = kwargs.pop('force_field', 'amber14-all.xml')
        water_model = kwargs.pop('water_model', 'amber14/tip3p.xml')
        cation = kwargs.pop('cation', 'Na+')
        anion = kwargs.pop('anion', 'Cl-')
        ion_conc = kwargs.pop('ion_conc', 0.15)
        # Model
        pdb = app.PDBFile(inpdb) 
        modeller = app.Modeller(pdb.topology, pdb.positions)
        forcefield = app.ForceField(force_field, water_model)
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)
        modeller.addIons(forcefield, cation, anion, ion_conc*mol/liter)
        # System
        mdsys = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds, )
        # Saving files
        with open(self.syspdb, 'w') as pdb_file:
            PDBFile.writeFile(modeller.topology, modeller.positions, pdb_file, keepIds=True)
        with open(self.sysxml, 'w') as xml_file:
            xml_file.write(mm.XmlSerializer.serialize(system))
        return mdsys

    def integ(self):
        # Create a Langevin integrator (300 K, friction 1/ps, time step 2 fs)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

        # Set up the simulation object
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # -----------------------
        # 2. Energy Minimization
        # -----------------------
        print('Minimizing energy...')
        simulation.minimizeEnergy()

        # -----------------------
        # 3. Equilibration Phase
        # -----------------------
        print('Starting equilibration...')
        simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
        equilibration_steps = 5000  # Adjust equilibration steps as needed
        simulation.step(equilibration_steps)
        print('Equilibration complete.')

        # Save the simulation state after equilibration
        print('Saving simulation state...')
        simulation.saveState('state.xml')

        # -----------------------
        # 4. Production Run (Resumed from saved state)
        # -----------------------
        # Later (or in a separate script), you can resume the simulation by reloading the state.
        print('Loading simulation state to continue production run...')
        simulation.loadState('state.xml')

        # Optionally, update or add reporters for the production run
        simulation.reporters = []  # Clear previous reporters if needed
        simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
        simulation.reporters.append(DCDReporter('trajectory_production.dcd', 1000))

        production_steps = 10000  # Adjust production steps as needed
        print('Starting production run...')
        simulation.step(production_steps)
        print('Production run complete.')


################################################################################
# MDRun class
################################################################################   

class MDRun(mmSystem):
    """
    Run molecular dynamics (MD) simulation using the specified input files.
    This method runs the MD simulation by calling an external simulation software, 
    such as GROMACS, with the provided input files.
    """
    
    def __init__(self, runname, *args, **kwargs):
        """
        Initializes required directories and files.
        
        Args:
            runname (str): 
            rundir (str): Directory for the system files.
            rmsname (str): Name of the system.
            cludir (str): clustering
            covdir (str): covariance analysis
            pngdir (str): figures
            kwargs: Additional keyword arguments.
        """
        super().__init__(*args)
        self.runname = runname
        # self.rundir = '/home/dyangali/tmp'
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, 'rms_analysis')
        self.covdir = os.path.join(self.rundir, 'cov_analysis')
        self.dddir  = os.path.join(self.rundir, 'dci_dfi')
        self.cludir = os.path.join(self.rundir, 'clusters')
        self.pngdir = os.path.join(self.rundir, 'png')
        
    def prepare_files(self):
        """
        Create necessary directories.
        """
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.dddir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)