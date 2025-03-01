"""File: gmxmd.py

Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations using GROMACS. The main
    classes include:

      - GmxSystem: Provides methods to prepare simulation files, process PDB
        files, run GROMACS commands, and perform various analyses on MD data.
      - MDRun: A subclass of GmxSystem dedicated to executing MD simulations and
        performing post-processing tasks (e.g., RMSF, RMSD, covariance analysis).

    Additionally, utility functions (e.g., sort_upper_lower_digit) are included
    to assist in organizing GROMACS multichain files.

Usage:
    Import this module and instantiate the GmxSystem or MDRun classes to set up
    and run your MD simulations.

Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - Pandas
    - GROMACS (with CLI tools such as gmx_mpi, gmx, etc.)
    - The reforge package and its dependencies

Author: DY
Date: 2025-02-27
"""

import os
import sys
import importlib.resources
import shutil
import subprocess as sp
import MDAnalysis as mda
import numpy as np
from reforge import cli, mdm, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger
from .mdsystem import MDSystem, MDRun


################################################################################
# GMX system class
################################################################################

class GmxSystem(MDSystem):
    """Class to set up and analyze protein-nucleotide-lipid systems for MD
    simulations using GROMACS.

    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """

    MDATDIR = importlib.resources.files("reforge") / "martini" / "data"
    MMDPDIR = importlib.resources.files("reforge") / "martini" / "data" / "mdp"
    MITPDIR = importlib.resources.files("reforge") / "martini" / "itp"

    def __init__(self, sysdir, sysname):
        super().__init__(sysdir, sysname)
        """Initializes the MD system with required directories and file paths.

        Parameters
        ----------
            sysdir (str): Base directory for the system files.
            sysname (str): Name of the MD system.

        Sets up paths for various files required for coarse-grained MD simulation.
        """
        self.sysgro = os.path.join(self.root, "system.gro")
        self.systop = os.path.join(self.root, "system.top")
        self.sysndx = os.path.join(self.root, "system.ndx")
        self.mdpdir = os.path.join(self.root, "mdp")

    def gmx(self, command="-h", clinput=None, clean_wdir=True, **kwargs):
        """Executes a GROMACS command using the reforge CLI.

        Parameters
        ----------
            command (str): The GROMACS command to run (default: "-h").
            clinput (str, optional): Input to pass to the command's stdin.
            clean_wdir (bool, optional): If True, cleans the working directory after execution.
            kwargs: Additional keyword arguments to pass to the CLI.

        Runs the command from the system's root directory.
        """
        with cd(self.root):
            cli.gmx(command, clinput=clinput, **kwargs)
            if clean_wdir:
                clean_dir()

    def prepare_files(self):
        """Prepares the simulation by creating necessary directories and copying input files.

        The method:
          - Creates directories for proteins, nucleotides, topologies, maps, mdp files,
            coarse-grained PDBs, GRO files, MD runs, data, and PNG outputs.
          - Copies .mdp files from the master MDP directory.
          - Copies 'water.gro' and 'atommass.dat' from the master data directory.
          - Copies .itp files from the master ITP directory to the system topology directory.
        """
        logger.info("Preparing files and directories")
        os.makedirs(self.prodir, exist_ok=True)
        os.makedirs(self.nucdir, exist_ok=True)
        os.makedirs(self.topdir, exist_ok=True)
        os.makedirs(self.mapdir, exist_ok=True)
        os.makedirs(self.mdpdir, exist_ok=True)
        os.makedirs(self.cgdir, exist_ok=True)
        os.makedirs(self.datdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        for file in os.listdir(self.MMDPDIR):
            if file.endswith(".mdp"):
                fpath = os.path.join(self.MMDPDIR, file)
                outpath = os.path.join(self.mdpdir, file)
                shutil.copy(fpath, outpath)
        shutil.copy(os.path.join(self.MDATDIR, "water.gro"), self.root)
        shutil.copy(os.path.join(self.MDATDIR, "atommass.dat"), self.root)
        for file in os.listdir(self.MITPDIR):
            if file.endswith(".itp"):
                fpath = os.path.join(self.MITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                shutil.copy(fpath, outpath)

    def make_system_top(self, add_resolved_ions=False, prefix="chain"):
        """Creates the system topology file by including all relevant ITP files and
        defining the system and molecule sections.

        Parameters
        ----------
            add_resolved_ions (bool, optional): If True, counts and includes resolved ions.
            prefix (str, optional): Prefix for ITP files to include (default: "chain").

        Writes the topology file (self.systop) with include directives and molecule counts.
        """
        logger.info("Writing system topology...")
        itp_files = [f for f in os.listdir(self.topdir) if f.startswith(prefix) and f.endswith(".itp")]
        itp_files = sort_upper_lower_digit(itp_files)
        with open(self.systop, "w") as f:
            # Include section
            f.write('#define GO_VIRT"\n')
            f.write("#define RUBBER_BANDS\n")
            f.write('#include "topol/martini_v3.0.0.itp"\n')
            f.write('#include "topol/martini_v3.0.0_rna.itp"\n')
            f.write('#include "topol/martini_ions.itp"\n')
            if "go_atomtypes.itp" in os.listdir(self.topdir):
                f.write('#include "topol/go_atomtypes.itp"\n')
                f.write('#include "topol/go_nbparams.itp"\n')
            f.write('#include "topol/martini_v3.0.0_solvents_v1.itp"\n')
            f.write('#include "topol/martini_v3.0.0_phospholipids_v1.itp"\n')
            f.write('#include "topol/martini_v3.0.0_ions_v1.itp"\n')
            f.write("\n")
            for filename in itp_files:
                f.write(f'#include "topol/{filename}"\n')
            # System name and molecule count
            f.write("\n[ system ]\n")
            f.write(f"Martini system for {self.sysname}\n")
            f.write("\n[molecules]\n")
            f.write("; name\t\tnumber\n")
            for filename in itp_files:
                molecule_name = os.path.splitext(filename)[0]
                f.write(f"{molecule_name}\t\t1\n")
            # Add resolved ions if requested.
            if add_resolved_ions:
                ions = self.count_resolved_ions()
                for ion, count in ions.items():
                    if count > 0:
                        f.write(f"{ion}    {count}\n")

    def make_gro_file(self, d=1.25, bt="dodecahedron"):
        """Generates the final GROMACS GRO file from coarse-grained PDB files.

        Parameters
        ----------
            d (float, optional): Distance parameter for the editconf command (default: 1.25).
            bt (str, optional): Box type for the editconf command (default: "dodecahedron").

        Converts PDB files to GRO files, merges them, and adjusts the system box.
        """
        with cd(self.root):
            cg_pdb_files = os.listdir(self.cgdir)
            cg_pdb_files = sort_upper_lower_digit(cg_pdb_files)
            for file in cg_pdb_files:
                if file.endswith(".pdb"):
                    pdb_file = os.path.join(self.cgdir, file)
                    gro_file = pdb_file.replace(".pdb", ".gro").replace("cgpdb", "gro")
                    command = f"gmx_mpi editconf -f {pdb_file} -o {gro_file}"
                    sp.run(command.split())
            # Merge all .gro files.
            gro_files = sorted(os.listdir(self.grodir))
            total_count = 0
            for filename in gro_files:
                if filename.endswith(".gro"):
                    filepath = os.path.join(self.grodir, filename)
                    with open(filepath, "r") as in_f:
                        atom_count = int(in_f.readlines()[1].strip())
                        total_count += atom_count
            with open(self.sysgro, "w") as out_f:
                out_f.write(f"{self.sysname} \n")
                out_f.write(f"  {total_count}\n")
                for filename in gro_files:
                    if filename.endswith(".gro"):
                        filepath = os.path.join(self.grodir, filename)
                        with open(filepath, "r") as in_f:
                            lines = in_f.readlines()[2:-1]
                            for line in lines:
                                out_f.write(line)
                out_f.write("10.00000   10.00000   10.00000\n")
            command = f"gmx_mpi editconf -f {self.sysgro} -d {d} -bt {bt}  -o {self.sysgro}"
            sp.run(command.split())

    def solvate(self, **kwargs):
        """Solvates the system using GROMACS solvate command.

        Parameters
        ----------
            kwargs: Additional parameters for the solvate command. Defaults:
                - cp: "solute.pdb"
                - cs: "water.gro"
        """
        kwargs.setdefault("cp", "solute.pdb")
        kwargs.setdefault("cs", "water.gro")
        self.gmx("solvate", p=self.systop, o=self.syspdb, **kwargs)

    def add_bulk_ions(self, solvent="W", **kwargs):
        """ Adds bulk ions to neutralize the system using GROMACS genion.

        Parameters
        ----------
        solvent (str, optional): 
            Solvent residue name (default: "W").
        kwargs (dict):
            Additional parameters for genion. Defaults include:
            - conc: 0.15
            - pname: "NA"
            - nname: "CL"
            - neutral: ""
        """
        kwargs.setdefault("conc", 0.15)
        kwargs.setdefault("pname", "NA")
        kwargs.setdefault("nname", "CL")
        kwargs.setdefault("neutral", "")
        self.gmx("grompp", f="mdp/ions.mdp", c=self.syspdb, p=self.systop, o="ions.tpr")
        self.gmx("genion", clinput="W\n", s="ions.tpr", p=self.systop, o=self.syspdb, **kwargs)
        self.gmx("editconf", f=self.syspdb, o=self.sysgro)
        clean_dir(self.root, "ions.tpr")

    def make_system_ndx(self, backbone_atoms=["CA", "P", "C1'"], water_resname='W'):
        """Creates an index (NDX) file for the system, separating solute, backbone, solvent, and individual chains.

        Parameters
        ----------
            backbone_atoms : list, optional
                List of atom names to include in the backbone (default: ["CA", "P", "C1'"]).
        """
        logger.info(f"Making index file from {self.syspdb}...")
        # self.syspdb = os.path.join(self.rootdir, 'sys.pdb')
        system = pdbtools.pdb2atomlist(self.syspdb)
        solute = pdbtools.pdb2atomlist(self.solupdb)
        solvent = AtomList(system[len(solute):])
        backbone = solute.mask(backbone_atoms, mode="name")
        not_water = system.mask_out(water_resname, mode='resname')
        system.write_ndx(self.sysndx, header="[ System ]", append=False, wrap=15) # 0
        solute.write_ndx(self.sysndx, header="[ Solute ]", append=True, wrap=15) # 1
        backbone.write_ndx(self.sysndx, header="[ Backbone ]", append=True, wrap=15) # 2
        solvent.write_ndx(self.sysndx, header="[ Solvent ]", append=True, wrap=15) # 3
        not_water.write_ndx(self.sysndx, header="[ Not_Water ]", append=True, wrap=15) # 4
        chids = sorted(set(solute.chids))
        for chid in chids:
            chain = solute.mask(chid, mode="chid")
            chain.write_ndx(self.sysndx, header=f"[ chain_{chid} ]", append=True, wrap=15)
        logger.info(f"Written index to {self.sysndx}")


    def initmd(self, runname):
        """Initializes a new GMX MD run.

        Parameters
        ----------
            runname (str): Name for the MD run.

        Returns:
            MDRun: An instance of the MDRun class for the specified run.
        """
        mdrun = MDRun(self.sysdir, self.sysname, runname)
        return mdrun


class GmxRun(GmxSystem, MDRun):
    """Subclass of GmxSystem for executing molecular dynamics (MD) simulations
    and performing post-processing analyses.
    """

    def __init__(self, sysdir, sysname, runname):
        """Initializes the MD run environment with additional directories for analysis.

        Parameters
        ----------
            sysdir (str): Base directory for the system.
            sysname (str): Name of the MD system.
            runname (str): Name for the MD run.
        """
        super().__init__(sysdir, sysname)
        self.runname = runname
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, "rms_analysis")
        self.covdir = os.path.join(self.rundir, "cov_analysis")
        self.lrtdir = os.path.join(self.rundir, "lrt_analysis")
        self.cludir = os.path.join(self.rundir, "clusters")
        self.pngdir = os.path.join(self.rundir, "png")
        self.str = os.path.join(self.rundir, "mdc.pdb")  # Structure file
        self.trj = os.path.join(self.rundir, "mdc.trr")  # Trajectory file

    def prepare_files(self):
        """Creates necessary directories for the MD run and copies essential files."""
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        shutil.copy("atommass.dat", os.path.join(self.rundir, "atommass.dat"))

    def empp(self, **kwargs):
        """Prepares the energy minimization run using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to "em.mdp" file.
                - c: Structure file.
                - r: Reference structure.
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("em.tpr").
        """
        kwargs.setdefault("f", os.path.join(self.mdpdir, "em.mdp"))
        kwargs.setdefault("c", self.sysgro)
        kwargs.setdefault("r", self.sysgro)
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "em.tpr")
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)

    def hupp(self, **kwargs):
        """Prepares the heat-up phase using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to "hu.mdp".
                - c: Starting structure ("em.gro").
                - r: Reference structure ("em.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("hu.tpr").
        """
        kwargs.setdefault("f", os.path.join(self.mdpdir, "hu.mdp"))
        kwargs.setdefault("c", "em.gro")
        kwargs.setdefault("r", "em.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "hu.tpr")
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)

    def eqpp(self, **kwargs):
        """Prepares the equilibration phase using GROMACS grompp.

        Parameters
        ----------
            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to "eq.mdp".
                - c: Starting structure ("hu.gro").
                - r: Reference structure ("hu.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("eq.tpr").
        """
        kwargs.setdefault("f", os.path.join(self.mdpdir, "eq.mdp"))
        kwargs.setdefault("c", "hu.gro")
        kwargs.setdefault("r", "hu.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "eq.tpr")
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)

    def mdpp(self, grompp=True, **kwargs):
        """Prepares the production MD run using GROMACS grompp.

        Parameters
        ----------
            grompp (bool, optional): Whether to run grompp (default: True).

            kwargs: Additional parameters for grompp. Defaults include:
                - f: Path to "md.mdp".
                - c: Starting structure ("eq.gro").
                - r: Reference structure ("eq.gro").
                - p: Topology file.
                - n: Index file.
                - o: Output TPR file ("md.tpr").
        """
        kwargs.setdefault("f", os.path.join(self.mdpdir, "md.mdp"))
        kwargs.setdefault("c", "eq.gro")
        kwargs.setdefault("r", "eq.gro")
        kwargs.setdefault("p", self.systop)
        kwargs.setdefault("n", self.sysndx)
        kwargs.setdefault("o", "md.tpr")
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)

    def mdrun(self, **kwargs):
        """Executes the production MD run using GROMACS mdrun.

        Parameters
        ----------
            kwargs: Additional parameters for mdrun. Defaults include:
                - deffnm: "md"
                - nsteps: "-2"
                - ntomp: "8"
        """
        kwargs.setdefault("deffnm", "md")
        kwargs.setdefault("nsteps", "-2")
        kwargs.setdefault("ntomp", "8")
        with cd(self.rundir):
            cli.gmx_mdrun(**kwargs)

    def trjconv(self, clinput=None, **kwargs):
        """Converts trajectories using GROMACS trjconv.

        Parameters
        ----------
            clinput (str, optional): Input to be passed to trjconv.
            kwargs: Additional parameters for trjconv.
        """
        with cd(self.rundir):
            cli.gmx_trjconv(clinput=clinput, **kwargs)

    def rmsf(self, clinput=None, **kwargs):
        """Calculates RMSF using GROMACS rmsf.

        Parameters
        ----------
            clinput (str, optional): Input for the rmsf command.
            kwargs: Additional parameters for rmsf. Defaults include:
                - s: Structure file.
                - f: Trajectory file.
                - o: Output xvg file.
        """
        xvg_file = os.path.join(self.rmsdir, "rmsf.xvg")
        npy_file = os.path.join(self.rmsdir, "rmsf.npy")
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("o", xvg_file)
        with cd(self.rmsdir):
            cli.gmx_rmsf(clinput=clinput, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[1])

    def rmsd(self, clinput=None, **kwargs):
        """Calculates RMSD using GROMACS rms.

        Parameters
        ----------
            clinput (str, optional): Input for the rms command.
            kwargs: Additional parameters for rms. Defaults include:
                - s: Structure file.
                - f: Trajectory file.
                - o: Output xvg file.
        """
        xvg_file = os.path.join(self.rmsdir, "rmsd.xvg")
        npy_file = os.path.join(self.rmsdir, "rmsd.npy")
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("o", xvg_file)
        with cd(self.rmsdir):
            cli.gmx_rms(clinput=clinput, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[0, 1])

    def rdf(self, clinput=None, **kwargs):
        """Calculates the radial distribution function using GROMACS rdf.

        Parameters
        ----------
            clinput (str, optional): Input for the rdf command.
            kwargs: Additional parameters for rdf. Defaults include:
                - f: Trajectory file.
                - s: Structure file.
                - n: Index file.
        """
        kwargs.setdefault("f", "mdc.xtc")
        kwargs.setdefault("s", "mdc.pdb")
        kwargs.setdefault("n", self.mdcndx)
        with cd(self.rmsdir):
            cli.gmx_rdf(clinput=clinput, **kwargs)

    def cluster(self, clinput=None, **kwargs):
        """Performs clustering using GROMACS cluster.

        Parameters
        ----------
            clinput (str, optional): Input for the clustering command.
            kwargs: Additional parameters for cluster.
        """
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        with cd(self.cludir):
            cli.gmx_cluster(clinput=clinput, **kwargs)

    def extract_cluster(self, clinput=None, **kwargs):
        """Extracts frames belonging to a cluster using GROMACS extract-cluster.

        Parameters
        ----------
            clinput (str, optional): Input for the extract-cluster command.
            kwargs: Additional parameters for extract-cluster. Defaults include:
            - clusters: "cluster.ndx"
        """
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("clusters", "cluster.ndx")
        with cd(self.cludir):
            cli.gmx_extract_cluster(clinput=clinput, **kwargs)

    def covar(self, clinput=None, **kwargs):
        """Calculates and diagonalizes the covariance matrix using GROMACS covar.

        Parameters
        ----------
            clinput (str, optional): Input for the covar command.
            kwargs: Additional parameters for covar. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
        """
        kwargs.setdefault("f", "../traj.xtc")
        kwargs.setdefault("s", "../traj.pdb")
        kwargs.setdefault("n", self.trjndx)
        with cd(self.covdir):
            cli.gmx_covar(self.covdir, clinput=clinput, **kwargs)

    def anaeig(self, clinput=None, **kwargs):
        r"""Analyzes eigenvectors using GROMACS anaeig.

        Parameters
        ----------
            clinput (str, optional): Input for the anaeig command.
            kwargs: Additional parameters for anaeig. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - v: Output eigenvector file.
        """
        kwargs.setdefault("f", "../traj.xtc")
        kwargs.setdefault("s", "../traj.pdb")
        kwargs.setdefault("v", "eigenvec.trr")
        cli.gmx_anaeig(self.covdir, clinput=clinput, **kwargs)

    def make_edi(self, clinput=None, **kwargs):
        """Prepares files for essential dynamics analysis using GROMACS make-edi.

        Parameters
        ----------
            clinput (str, optional): Input for the make-edi command.
            kwargs: Additional parameters for make-edi. Defaults include:
            - f: Eigenvector file.
            - s: Structure file.
        """
        kwargs.setdefault("f", "eigenvec.trr")
        kwargs.setdefault("s", "../traj.pdb")
        cli.gmx_make_edi(self.covdir, clinput=clinput, **kwargs)

    def get_rmsf_by_chain(self, **kwargs):
        """Calculates RMSF for each chain in the system using GROMACS rmsf.

        Parameters
        ----------
            kwargs: Additional parameters for the rmsf command. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
            - res: Whether to output per-residue RMSF (default: "no").
            - fit: Whether to fit the trajectory (default: "yes").
        """
        kwargs.setdefault("f", "traj.xtc")
        kwargs.setdefault("s", "traj.pdb")
        kwargs.setdefault("n", self.trjndx)
        kwargs.setdefault("res", "no")
        kwargs.setdefault("fit", "yes")
        for idx, chain in enumerate(self.chains):
            idx = idx + 1
            cli.gmx_rmsf(
                self.rundir,
                clinput=f"{idx}\n{idx}\n",
                o=os.path.join(self.rmsdir, f"rmsf_{chain}.xvg"),
                **kwargs,
            )

    def get_rmsd_by_chain(self, **kwargs):
        """Calculates RMSD for each chain in the system using GROMACS rmsd.

        Parameters
        ----------
            kwargs: Additional parameters for the rmsd command. Defaults include:
            - f: Trajectory file.
            - s: Structure file.
            - n: Index file.
        """
        kwargs.setdefault("f", "traj.xtc")
        kwargs.setdefault("s", "traj.pdb")
        kwargs.setdefault("n", self.trjndx)
        for idx, chain in enumerate(self.chains):
            idx = idx + 1
            cli.gmx_rmsf(
                self.rundir,
                clinput=f"{idx}\n",
                o=os.path.join(self.rmsdir, f"rmsf_{chain}.xvg"),
                **kwargs,
            )


