"""
Module mmmd
===========

This module defines classes and functions to set up and run coarse-grained (CG)
molecular dynamics simulations using OpenMM and associated tools. It includes
the system preparation, file management, and MD run routines.

Classes
-------
MmSystem
    Sets up and analyzes protein-nucleotide-lipid systems for MD simulations.
MdRun
    Inherits from MmSystem to perform an MD simulation run, including energy
    minimization, equilibration, and production.
    
Helper Functions
----------------
sort_uld(alist)
    Sorts characters in a list so that uppercase letters come first, then lowercase, 
    followed by digits.
"""

import os
import sys
import importlib.resources
import shutil

import openmm as mm
from openmm import app
from openmm.unit import nanometer, molar

# Local imports from reforge (only the needed modules)
from reforge import pdbtools


################################################################################
# CG system class
################################################################################

class MmSystem:
    """Set up and analyze protein–nucleotide–lipid systems for CG MD simulation.

    This class initializes file paths and directories required for running CG
    molecular dynamics simulations.

    Parameters
    ----------
    sysdir : str
        Directory for the system files.
    sysname : str
        Name of the system.
    **kwargs : dict, optional
        Additional keyword arguments (currently unused).

    Attributes
    ----------
    sysname : str
        Name of the system.
    sysdir : str
        Absolute path to the system directory.
    wdir : str
        Working directory (sysdir joined with sysname).
    inpdb : str
        Path to the input PDB file.
    syspdb : str
        Path to the system PDB file.
    sysxml : str
        Path to the system XML file.
    mdcpdb : str
        Path to the MD configuration PDB file.
    trjpdb : str
        Path to the trajectory PDB file.
    prodir : str
        Directory for protein files.
    nucdir : str
        Directory for nucleotide files.
    iondir : str
        Directory for ion files.
    ionpdb : str
        Path to the ion PDB file.
    topdir : str
        Directory for topology files.
    mapdir : str
        Directory for mapping files.
    mdpdir : str
        Directory for MD parameter files.
    cgdir : str
        Directory for CG PDB files.
    mddir : str
        Directory for MD run files.
    datdir : str
        Directory for data files.
    pngdir : str
        Directory for PNG figures.
    _chains : list
        Cached list of chain identifiers.
    _mdruns : list
        Cached list of MD run directories.
    """

    MDATDIR = importlib.resources.files("reforge") / "martini" / "data"
    MITPDIR = importlib.resources.files("reforge") / "martini" / "itp"
    NUC_RESNAMES = [
        "A",
        "C",
        "G",
        "U",
        "RA3",
        "RA5",
        "RC3",
        "RC5",
        "RG3",
        "RG5",
        "RU3",
        "RU5",
    ]

    def __init__(self, sysdir, sysname, **kwargs):
        """Initialize the MD system with required directories and file paths.

        Parameters
        ----------
        sysdir : str
            Directory for the system files.
        sysname : str
            Name of the system.
        **kwargs : dict, optional
            Additional keyword arguments.
            
        Notes
        -----
        Sets up paths to various files required for CG MD simulation.
        """
        self.sysname = sysname
        self.sysdir = os.path.abspath(sysdir)
        self.wdir = os.path.join(self.sysdir, sysname)
        self.inpdb = os.path.join(self.wdir, "inpdb.pdb")
        self.syspdb = os.path.join(self.wdir, "system.pdb")
        self.sysxml = os.path.join(self.wdir, "system.xml")
        self.mdcpdb = os.path.join(self.wdir, "mdc.pdb")
        self.trjpdb = os.path.join(self.wdir, "traj.pdb")
        self.prodir = os.path.join(self.wdir, "proteins")
        self.nucdir = os.path.join(self.wdir, "nucleotides")
        self.iondir = os.path.join(self.wdir, "ions")
        self.ionpdb = os.path.join(self.iondir, "ions.pdb")
        self.topdir = os.path.join(self.wdir, "topol")
        self.mapdir = os.path.join(self.wdir, "map")
        self.mdpdir = os.path.join(self.wdir, "mdp")
        self.cgdir = os.path.join(self.wdir, "cgpdb")
        self.mddir = os.path.join(self.wdir, "mdruns")
        self.datdir = os.path.join(self.wdir, "data")
        self.pngdir = os.path.join(self.wdir, "png")
        self._chains = []
        self._mdruns = []

    @property
    def chains(self):
        """Retrieve the chain identifiers from the system PDB file.

        Returns
        -------
        list
            List of chain identifiers.
            
        Notes
        -----
        If chain IDs have already been extracted, returns the cached list.
        Otherwise, parses the system PDB file.
        """
        if self._chains:
            return self._chains
        chain_names = set()
        with open(self.syspdb, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21].strip()  # Chain ID is at column 22 (index 21)
                    if chain_id:
                        chain_names.add(chain_id)
        self._chains = sort_uld(chain_names)
        return self._chains

    @chains.setter
    def chains(self, chains):
        self._chains = chains

    @property
    def mdruns(self):
        """Retrieve the list of MD run directories.

        Returns
        -------
        list
            List of MD run directory names.
            
        Notes
        -----
        If the list is already cached, returns it. Otherwise, looks up the directories
        in the MD runs folder.
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
        """Prepare simulation directories and copy necessary input files.

        Notes
        -----
        Creates directories (proteins, nucleotides, topology, mapping, MD parameters,
        CG PDB, MD runs, data, PNG) and copies files from source directories.
        """
        print("Preparing files and directories", file=sys.stderr)
        os.makedirs(self.prodir, exist_ok=True)
        os.makedirs(self.nucdir, exist_ok=True)
        os.makedirs(self.topdir, exist_ok=True)
        os.makedirs(self.mapdir, exist_ok=True)
        os.makedirs(self.mdpdir, exist_ok=True)
        os.makedirs(self.cgdir, exist_ok=True)
        os.makedirs(self.wdir, exist_ok=True)  # In case working directory is not present
        os.makedirs(self.datdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        # Create additional directory (e.g. grodir) if needed
        grodir = os.path.join(self.wdir, "gro")
        os.makedirs(grodir, exist_ok=True)
        for file in os.listdir(self.MDATDIR):
            if file.endswith(".mdp"):
                fpath = os.path.join(self.MDATDIR, file)
                outpath = os.path.join(self.mdpdir, file)
                shutil.copy(fpath, outpath)
        shutil.copy(os.path.join(self.MDATDIR, "water.gro"), self.wdir)
        for file in os.listdir(self.MITPDIR):
            if file.endswith(".itp"):
                fpath = os.path.join(self.MITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                shutil.copy(fpath, outpath)

    def clean_pdb(self, pdb_file, **kwargs):
        """Clean the starting PDB file using PDBfixer by OpenMM.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file.
        **kwargs : dict, optional
            Additional keyword arguments for the cleaning routine.
        """
        print("Cleaning the PDB", file=sys.stderr)
        pdbtools.clean_pdb(pdb_file, self.inpdb, **kwargs)

    @staticmethod
    def forcefield(force_field="amber14-all.xml", water_model="amber14/tip3p.xml", **kwargs):
        """Create and return an OpenMM ForceField object.

        Parameters
        ----------
        force_field : str, optional
            Force field file or identifier (default: 'amber14-all.xml').
        water_model : str, optional
            Water model file or identifier (default: 'amber14/tip3p.xml').
        **kwargs : dict, optional
            Additional keyword arguments for the ForceField constructor.

        Returns
        -------
        openmm.app.ForceField
            The constructed ForceField object.
        """
        forcefield = app.ForceField(force_field, water_model, **kwargs)
        return forcefield

    @staticmethod
    def modeller(inpdb, forcefield, **kwargs):
        """Generate a modeller object with added solvent.

        Parameters
        ----------
        inpdb : str
            Path to the input PDB file.
        forcefield : openmm.app.ForceField
            The force field object to be used.
        **kwargs : dict, optional
            Additional keyword arguments for solvent addition. Default values include:
                model : 'tip3p'
                boxShape : 'dodecahedron'
                padding : 1.0 * nanometer
                positiveIon : 'Na+'
                negativeIon : 'Cl-'
                ionicStrength : 0.1 * molar

        Returns
        -------
        openmm.app.Modeller
            The modeller object with solvent added.
        """
        kwargs.setdefault("model", "tip3p")
        kwargs.setdefault("boxShape", "dodecahedron")
        kwargs.setdefault("padding", 1.0 * nanometer)
        kwargs.setdefault("positiveIon", "Na+")
        kwargs.setdefault("negativeIon", "Cl-")
        kwargs.setdefault("ionicStrength", 0.1 * molar)
        pdb_file = app.PDBFile(inpdb)
        modeller_obj = app.Modeller(pdb_file.topology, pdb_file.positions)
        modeller_obj.addSolvent(forcefield, **kwargs)
        return modeller_obj

    def model(self, forcefield, modeller_obj, barostat=None, thermostat=None, **kwargs):
        """Create a simulation model using the specified force field and modeller.

        Parameters
        ----------
        forcefield : openmm.app.ForceField
            The force field object.
        modeller_obj : openmm.app.Modeller
            The modeller object with the prepared topology.
        barostat : openmm.Force, optional
            Barostat force to add (default: None).
        thermostat : openmm.Force, optional
            Thermostat force to add (default: None).
        **kwargs : dict, optional
            Additional keyword arguments for creating the system. Defaults include:
                nonbondedMethod : app.PME
                nonbondedCutoff : 1.0 * nanometer
                constraints : app.HBonds

        Returns
        -------
        openmm.System
            The simulation system created by the force field.
        """
        kwargs.setdefault("nonbondedMethod", app.PME)
        kwargs.setdefault("nonbondedCutoff", 1.0 * nanometer)
        kwargs.setdefault("constraints", app.HBonds)
        model_obj = forcefield.createSystem(modeller_obj.topology, **kwargs)
        if barostat:
            model_obj.addForce(barostat)
        if thermostat:
            model_obj.addForce(thermostat)
        with open(self.syspdb, "w", encoding="utf-8") as file:
            app.PDBFile.writeFile(modeller_obj.topology, modeller_obj.positions, file, keepIds=True)
        with open(self.sysxml, "w", encoding="utf-8") as file:
            file.write(mm.XmlSerializer.serialize(model_obj))
        return model_obj


################################################################################
# MDRun class
################################################################################

class MdRun(MmSystem):
    """Run molecular dynamics (MD) simulation using specified input files.

    This class extends MmSystem to perform an MD simulation run including
    energy minimization, equilibration, and production.

    Parameters
    ----------
    sysdir : str
        Directory for the system files.
    sysname : str
        Name of the system.
    runname : str
        Name of the MD run.

    Attributes
    ----------
    rundir : str
        Directory for the run files.
    rmsdir : str
        Directory for RMS analysis.
    covdir : str
        Directory for covariance analysis.
    lrtdir : str
        Directory for LRT analysis.
    cludir : str
        Directory for clustering.
    pngdir : str
        Directory for output figures.
    """

    def __init__(self, sysdir, sysname, runname):
        """Initialize the MD run with required directories and files.

        Parameters
        ----------
        sysdir : str
            Directory for the system files.
        sysname : str
            Name of the system.
        runname : str
            Name of the MD run.
        """
        super().__init__(sysdir, sysname)
        self.runname = runname
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, "rms_analysis")
        self.covdir = os.path.join(self.rundir, "cov_analysis")
        self.lrtdir = os.path.join(self.rundir, "lrt_analysis")
        self.cludir = os.path.join(self.rundir, "clusters")
        self.pngdir = os.path.join(self.rundir, "png")

    def prepare_files(self):
        """Create directories required for the MD run.

        Notes
        -----
        Creates directories for run outputs including RMS, covariance, LRT,
        clustering, and figures.
        """
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)

    def build_modeller(self):
        """Generate a modeller object from the system PDB file.

        Returns
        -------
        openmm.app.Modeller
            The modeller object initialized with the system topology and positions.
        """
        pdb_file = app.PDBFile(self.syspdb)
        modeller_obj = app.Modeller(pdb_file.topology, pdb_file.positions)
        return modeller_obj

    def simulation(self, modeller_obj, integrator):
        """Initialize and return a simulation object for the MD run.

        Parameters
        ----------
        modeller_obj : openmm.app.Modeller
            The modeller object with prepared topology and positions.
        integrator : openmm.Integrator
            The integrator for the simulation.

        Returns
        -------
        openmm.app.Simulation
            The initialized simulation object.
        """
        simulation_obj = app.Simulation(modeller_obj.topology, self.sysxml, integrator)
        simulation_obj.context.setPositions(modeller_obj.positions)
        return simulation_obj

    def save_state(self, simulation_obj, file_prefix="sim"):
        """Save the current simulation state to XML and PDB files.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        file_prefix : str, optional
            Prefix for the state files (default: 'sim').

        Notes
        -----
        Saves the simulation state as an XML file and writes the current positions to a PDB file.
        """
        pdb_file = os.path.join(self.rundir, file_prefix + ".pdb")
        xml_file = os.path.join(self.rundir, file_prefix + ".xml")
        simulation_obj.saveState(xml_file)
        state = simulation_obj.context.getState(getPositions=True)
        positions = state.getPositions()
        with open(pdb_file, "w", encoding="utf-8") as file:
            app.PDBFile.writeFile(simulation_obj.topology, positions, file, keepIds=True)

    def em(self, simulation_obj, tolerance=100, max_iterations=1000):
        """Perform energy minimization for the simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        tolerance : float, optional
            Tolerance for energy minimization (default: 100).
        max_iterations : int, optional
            Maximum number of iterations (default: 1000).

        Notes
        -----
        Minimizes the energy, saves the minimized state, and logs progress.
        """
        print("Minimizing energy...", file=sys.stderr)
        log_file = os.path.join(self.rundir, "em.log")
        reporter = app.StateDataReporter(
            log_file, 100, step=True, potentialEnergy=True, temperature=True
        )
        simulation_obj.reporters.append(reporter)
        simulation_obj.minimizeEnergy(tolerance, max_iterations)
        self.save_state(simulation_obj, "em")
        print("Minimization complete.", file=sys.stderr)

    def eq(self, simulation_obj, nsteps=10000, nlog=10000, **kwargs):
        """Run equilibration simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of steps for equilibration (default: 10000).
        nlog : int, optional
            Logging frequency (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments for the StateDataReporter.

        Notes
        -----
        Loads the minimized state, runs equilibration, and saves the equilibrated state.
        """
        print("Starting equilibration...")
        kwargs.setdefault("step", True)
        kwargs.setdefault("potentialEnergy", True)
        kwargs.setdefault("temperature", True)
        kwargs.setdefault("density", True)
        em_xml = os.path.join(self.rundir, "em.xml")
        log_file = os.path.join(self.rundir, "eq.log")
        reporter = app.StateDataReporter(log_file, nlog, **kwargs)
        simulation_obj.loadState(em_xml)
        simulation_obj.reporters.append(reporter)
        simulation_obj.step(nsteps)
        self.save_state(simulation_obj, "eq")
        print("Equilibration complete.")

    def md(self, simulation_obj, nsteps=100000, nout=1000, nlog=10000, nchk=10000, **kwargs):
        """Run production MD simulation.

        Parameters
        ----------
        simulation_obj : openmm.app.Simulation
            The simulation object.
        nsteps : int, optional
            Number of production steps (default: 100000).
        nout : int, optional
            Frequency for writing trajectory frames (default: 1000).
        nlog : int, optional
            Logging frequency (default: 10000).
        nchk : int, optional
            Checkpoint frequency (default: 10000).
        **kwargs : dict, optional
            Additional keyword arguments for the StateDataReporter.

        Notes
        -----
        Loads the equilibrated state, runs production, and saves the final simulation state.
        """
        print("Production run...")
        kwargs.setdefault("step", True)
        kwargs.setdefault("time", True)
        kwargs.setdefault("potentialEnergy", True)
        kwargs.setdefault("temperature", True)
        kwargs.setdefault("density", False)
        eq_xml = os.path.join(self.rundir, "eq.xml")
        trj_file = os.path.join(self.rundir, "md.dcd")
        log_file = os.path.join(self.rundir, "md.log")
        xml_file = os.path.join(self.rundir, "md.xml")
        pdb_file = os.path.join(self.rundir, "md.pdb")
        trj_reporter = app.DCDReporter(trj_file, nout, append=False)
        log_reporter = app.StateDataReporter(log_file, nlog, **kwargs)
        xml_reporter = app.CheckpointReporter(xml_file, nchk, writeState=True)
        # If PDBReporter is not used, remove the assignment to avoid unused variable warnings.
        reporters = [trj_reporter, log_reporter, xml_reporter]
        simulation_obj.loadState(eq_xml)
        simulation_obj.reporters.extend(reporters)
        simulation_obj.step(nsteps)
        self.save_state(simulation_obj, "md")
        print("Production complete.")


################################################################################
# Helper functions
################################################################################

def sort_uld(alist):
    """Sort characters in a list in a specific order.

    Parameters
    ----------
    alist : list of str
        List of characters to sort.

    Returns
    -------
    list of str
        Sorted list with uppercase letters first, then lowercase letters, and digits last.

    Notes
    -----
    This function is used to help organize GROMACS multichain files.
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist
