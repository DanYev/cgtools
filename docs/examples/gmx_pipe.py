#!/usr/bin/env python
"""
Gmx Pipe Tutorial
=================

This module serves as a tutorial for setting up and running coarse-grained
molecular dynamics (MD) simulations using the reForge package and GROMACS. It
provides a pipeline that performs several key tasks:
System Setup: Copying force field and parameter files, preparing directories,
cleaning and sorting the input PDB file, and splitting chains.
Coarse-Graining: Applying coarse-graining methods for proteins (using both
Go-model and elastic network approaches) and nucleotides.
Solvation and Ion Addition: Solvating the system and adding ions to neutralize it.
MD Simulation: Running energy minimization, equilibration, and production MD runs.
Post-Processing: Converting trajectories, and performing
analyses such as RMSF, RMSD, covariance analysis, clustering, and time-dependent
correlation calculations.

Author: DY
Date: 2025-XX-XX
"""

import os
import sys
import numpy as np
import pandas as pd
import shutil
import MDAnalysis as mda
from pathlib import Path
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import gmxSystem, MDRun
from reforge.utils import *  # Assuming this imports required utilities

#%%
def setup(*args):
    """Runs the complete setup pipeline.

    This function calls the appropriate setup function for the system.

    Args:
        *args: Positional arguments passed to the setup functions.
    """
    setup_cg_protein_rna(*args)
    # Uncomment the following line if you also want to setup a protein-membrane system.
    # setup_cg_protein_membrane(*args)

# Some comments
#%%
def setup_cg_protein_rna(sysdir, sysname):
    """Sets up a coarse-grained protein/RNA system.

    This function performs the following steps:
        1. Prepares system files and directories.
        2. Sorts and cleans the input PDB file.
        3. Splits chains and cleans them using GROMACS.
        4. Retrieves GO contact maps.
        5. Applies coarse-graining (using the Go-model for proteins and Martini RNA for nucleotides).
        6. Creates topology and structure files.
        7. Solvates the system and adds ions.
        8. Generates index files.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
    """
    mdsys = gmxSystem(sysdir, sysname)
    # 1. Prepare force field, parameter files, and directories.
    mdsys.prepare_files()
    mdsys.sort_input_pdb(f"{sysname}.pdb")

    # 2. Clean the PDB and split chains using GROMACS.
    mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput="8\n 7\n", ignh="no", renum="yes")
    mdsys.split_chains()
    mdsys.clean_chains_gmx(clinput="8\n 7\n", ignh="yes", renum="yes")
    mdsys.get_go_maps(append=True)

    # 3. Coarse-grain the proteins and RNA.
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)
    mdsys.martinize_rna(ef=200, el=0.3, eu=1.2, p="backbone", pf=500, append=True)
    mdsys.make_martini_topology_file(add_resolved_ions=False, prefix="chain")
    mdsys.make_cgpdb_file(bt="dodecahedron", d="1.2")
    
    # 4. Solvate and add ions.
    solvent = os.path.join(mdsys.wdir, "water.gro")
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
    mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")
    
    # 5. Create index files.
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


def setup_cg_protein_membrane(sysdir, sysname):
    """Sets up a coarse-grained protein/membrane system.

    This function is similar to setup_cg_protein_rna but is intended for systems
    that include a lipid bilayer.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
    """
    mdsys = gmxSystem(sysdir, sysname)
    mdsys.prepare_files()
    mdsys.sort_input_pdb(f"{sysname}.pdb")
    mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput="8\n 7\n", ignh="no", renum="yes")
    mdsys.split_chains()
    mdsys.clean_chains_gmx(clinput="8\n 7\n", ignh="yes", renum="yes")
    mdsys.get_go_maps(append=True)
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=True)
    mdsys.martinize_rna(ef=200, el=0.3, eu=1.2, p="backbone", pf=500, append=True)
    mdsys.make_martini_topology_file(add_resolved_ions=False, prefix="chain")
    mdsys.make_cgpdb_file(bt="dodecahedron", d="1.2")
    solvent = os.path.join(mdsys.wdir, "water.gro")
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
    mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


def md(sysdir, sysname, runname, ntomp):
    """Runs a complete MD simulation.

    This function performs an MD simulation by executing the energy minimization,
    equilibration, and production MD steps in sequence.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.
        ntomp (int): Number of CPU threads to use.
    """
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    # Choose appropriate mdp files.
    em_mdp = os.path.join(mdrun.mdpdir, "em.mdp")
    eq_mdp = os.path.join(mdrun.mdpdir, "eq.mdp")
    md_mdp = os.path.join(mdrun.mdpdir, "md.mdp")
    mdrun.empp(f=em_mdp)
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.eqpp(f=eq_mdp, c="em.gro", r="em.gro", maxwarn=10)
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=md_mdp, c="eq.gro", r="eq.gro")
    mdrun.mdrun(deffnm="md", ntomp=ntomp)


def extend(sysdir, sysname, runname, ntomp):
    """Extends an ongoing MD simulation.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name of the MD run to extend.
        ntomp (int): Number of CPU threads to use.
    """
    mdsys = gmxSystem(sysdir, sysname)
    mdrun = mdsys.initmd(runname)
    mdrun.mdrun(deffnm="md", cpi="md.cpt", ntomp=ntomp, nsteps=-2)


def make_ndx(sysdir, sysname, **kwargs):
    """Generates index (NDX) files for the system.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        **kwargs: Additional keyword arguments for index file generation.
    """
    mdsys = gmxSystem(sysdir, sysname)
    mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


def trjconv(sysdir, sysname, runname, mode="solu", fit="rot+trans", **kwargs):
    """Converts trajectory files using GROMACS trjconv.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.
        mode (str, optional): Mode to use for conversion ("solu" or "bb").
        fit (str, optional): Fitting method to apply (default: "rot+trans").
        **kwargs: Additional parameters for trjconv.
    """
    kwargs.setdefault("b", 0)      # in ps
    kwargs.setdefault("dt", 1000)  # in ps
    kwargs.setdefault("e", 1000000)  # in ps
    mdrun = MDRun(sysdir, sysname, runname)
    if mode == "solu":
        k = 1
    elif mode == "bb":
        k = 2
    if fit:
        mdrun.trjconv(clinput="0\n0\n", s="mdc.pdb", f="mdc.trr", o="mdv.pdb", dt=10000)
    clean_dir(mdrun.rundir)


def rms_analysis(sysdir, sysname, runname, **kwargs):
    """Performs RMSF and RMSD analysis on the MD run.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.
        **kwargs: Additional parameters for analysis (e.g., time range, dt).

    This function calculates RMSF and RMSD using GROMACS commands.
    """
    kwargs.setdefault("b", 50000)   # in ps
    kwargs.setdefault("dt", 1000)     # in ps
    kwargs.setdefault("e", 10000000)  # in ps
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.rmsf(clinput="2\n 2\n", s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit="yes", res="yes", **kwargs)
    mdrun.rmsd(clinput="2\n 2\n", s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit="rot+trans", **kwargs)


def cluster(sysdir, sysname, runname, **kwargs):
    """Performs clustering analysis on the MD run.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.
        **kwargs: Additional parameters for clustering (e.g., cutoff, method).

    This function runs GROMACS clustering and extracts clusters.
    """
    mdrun = MDRun(sysdir, sysname, runname)
    b = 100000
    mdrun.cluster(clinput="1\n 1\n", b=b, dt=1000, cutoff=0.15, method="gromos",
                  cl="clusters.pdb", clndx="cluster.ndx", av="yes")
    mdrun.extract_cluster()


def cov_analysis(sysdir, sysname, runname):
    """Performs covariance and perturbation matrix analysis on the MD run.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.

    This function computes covariance matrices, perturbation matrices,
    and calculates DFI/DCI values.
    """
    mdrun = MDRun(sysdir, sysname, runname)
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms.select_atoms("name CA or name P or name C1'")
    if not ag:
        ag = u.atoms.select_atoms("name BB or name BB1 or name BB3")
    mdrun.get_covmats(u, ag, sample_rate=1, b=50000, e=1000000, n=4, outtag="covmat")
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag="dfi")
    mdrun.get_dci(outtag="dci", asym=False)
    mdrun.get_dci(outtag="asym", asym=True)


def tdlrt_analysis(sysdir, sysname, runname):
    """Performs time-dependent correlation analysis on the MD run.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.

    This function calculates the position-velocity correlation function using a GPU method.
    """
    mdrun = MDRun(sysdir, sysname, runname)
    b = 0
    e = 100000
    sample_rate = 1
    ntmax = 1000
    tag = "pv"
    corr_file = os.path.join(mdrun.lrtdir, f"corr_{tag}.npy")
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms
    positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e)
    velocities = io.read_velocities(u, ag, sample_rate=sample_rate, b=b, e=e)
    corr = mdm.ccf(positions, velocities, ntmax=ntmax, n=5, mode="gpu", center=True)
    np.save(corr_file, corr)


def get_averages(sysdir, sysname):
    """Calculates averages from various analysis files.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.

    This function calculates mean and error values for perturbation and covariance matrices.
    """
    mdsys = gmxSystem(sysdir, sysname)
    mdsys.get_mean_sem(pattern="pertmat*.npy")
    mdsys.get_mean_sem(pattern="covmat*.npy")
    # Uncomment the following lines if needed:
    # mdsys.get_mean_sem(pattern="dfi*.npy")
    # mdsys.get_mean_sem(pattern="dci*.npy")
    # mdsys.get_mean_sem(pattern="asym*.npy")
    # mdsys.get_mean_sem(pattern="rmsf*.npy")


def get_td_averages(sysdir, sysname, loop=True, fname="corr_pv.npy"):
    """Calculates time-dependent averages from analysis files.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        loop (bool, optional): If True, processes files sequentially. Default is True.
        fname (str, optional): Filename pattern to average (default: "corr_pv.npy").

    Returns:
        numpy.ndarray: The time-dependent average.
    """
    mdsys = gmxSystem(sysdir, sysname)
    print("Getting averages", file=sys.stderr)
    files = io.pull_files(mdsys.mddir, fname)
    if loop:
        print(f"Processing {files[0]}", file=sys.stderr)
        average = np.load(files[0])
        for f in files[1:]:
            print(f"Processing {f}", file=sys.stderr)
            arr = np.load(f)
            average += arr
        average /= len(files)
    else:
        arrays = [np.load(f) for f in files]
        average = np.average(arrays, axis=0)
    np.save(os.path.join(mdsys.datdir, fname), average)
    print("Done!", file=sys.stderr)
    return average


def test(sysdir, sysname, runname, **kwargs):
    """A simple test function for the pipeline.

    Args:
        sysdir (str): Base directory for the system.
        sysname (str): Name of the MD system.
        runname (str): Name for the MD run.
        **kwargs: Additional arguments (unused).

    Prints a success message.
    """
    print("passed", file=sys.stderr)


if __name__ == "__main__":
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
    if command in commands:
        commands[command](*args)
    else:
        raise ValueError(f"Unknown command: {command}")
