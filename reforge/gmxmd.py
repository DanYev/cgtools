"""File: gmxmd.py
Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations using GROMACS. The main
    classes include:

      - GmxSystem: Provides methods to prepare simulation files, process PDB
                   files, run GROMACS commands, and perform various analyses on
                   MD data.
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


################################################################################
# GMX system class
################################################################################


class GmxSystem:
    """Class to set up and analyze protein-nucleotide-lipid systems for MD
    simulations using GROMACS.

    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """

    MDATDIR = importlib.resources.files("reforge") / "martini" / "data"
    MMDPDIR = importlib.resources.files("reforge") / "martini" / "data" / "mdp"
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

    def __init__(self, sysdir, sysname):
        """Initializes the MD system with required directories and file paths.

        Args:
            sysdir (str): Base directory for the system files.
            sysname (str): Name of the MD system.
            kwargs: Additional keyword arguments (currently unused).

        Sets up paths for various files required for coarse-grained MD simulation.
        """
        self.sysname = sysname
        self.sysdir = os.path.abspath(sysdir)
        self.root = os.path.join(self.sysdir, sysname)
        self.inpdb = os.path.join(self.root, "inpdb.pdb")
        self.solupdb = os.path.join(self.root, "solute.pdb")
        self.syspdb = os.path.join(self.root, "system.pdb")
        self.sysgro = os.path.join(self.root, "system.gro")
        self.systop = os.path.join(self.root, "system.top")
        self.sysndx = os.path.join(self.root, "system.ndx")
        self.mdcpdb = os.path.join(self.root, "mdc.pdb")
        self.mdcndx = os.path.join(self.root, "mdc.ndx")
        self.bbndx = os.path.join(self.root, "bb.ndx")
        self.trjpdb = os.path.join(self.root, "traj.pdb")
        self.trjndx = os.path.join(self.root, "traj.ndx")
        self.prodir = os.path.join(self.root, "proteins")
        self.nucdir = os.path.join(self.root, "nucleotides")
        self.iondir = os.path.join(self.root, "ions")
        self.ionpdb = os.path.join(self.iondir, "ions.pdb")
        self.topdir = os.path.join(self.root, "topol")
        self.mapdir = os.path.join(self.root, "map")
        self.mdpdir = os.path.join(self.root, "mdp")
        self.cgdir = os.path.join(self.root, "cgpdb")
        self.grodir = os.path.join(self.root, "gro")
        self.mddir = os.path.join(self.root, "mdruns")
        self.datdir = os.path.join(self.root, "data")
        self.pngdir = os.path.join(self.root, "png")

    @property
    def chains(self):
        """Retrieves and returns a sorted list of chain identifiers from the
        input PDB.

        Returns:
            list: Sorted chain identifiers extracted from the PDB file.
        """
        atoms = io.pdb2atomlist(self.inpdb)
        chains = sort_upper_lower_digit(set(atoms.chids))
        return chains

    def gmx(self, command="-h", clinput=None, clean_wdir=True, **kwargs):
        """Executes a GROMACS command using the reforge CLI.

        Args:
            command (str): The GROMACS command to run (default: '-h').
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
        """Prepares the simulation by creating necessary directories and
        copying input files.

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
        os.makedirs(self.grodir, exist_ok=True)
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

    def sort_input_pdb(self, in_pdb="inpdb.pdb"):
        """Sorts and renames atoms and chains in the input PDB file.

        Args:
            in_pdb (str): Path to the input PDB file (default: 'inpdb.pdb').

        Uses pdbtools to perform sorting and renaming, saving the result to the
        designated input PDB path.
        """
        with cd(self.root):
            pdbtools.sort_pdb(in_pdb, self.inpdb)

    def clean_pdb_mm(self, in_pdb=None, **kwargs):
        """Cleans the starting PDB file using PDBfixer (via OpenMM).

        Args:
            in_pdb (str, optional): Input PDB file to clean. If None, uses self.inpdb.
            kwargs: Additional keyword arguments for pdbtools.clean_pdb.
        """
        logger.info("Cleaning the PDB using OpenMM's PDBfixer...")
        if not in_pdb:
            in_pdb = self.inpdb
        pdbtools.clean_pdb(in_pdb, in_pdb, **kwargs)

    def clean_pdb_gmx(self, in_pdb=None, **kwargs):
        """Cleans the PDB file using GROMACS pdb2gmx tool.

        Args:
            in_pdb (str, optional): Input PDB file to clean. If None, uses self.inpdb.
            kwargs: Additional keyword arguments for the GROMACS command.

        After running pdb2gmx, cleans up temporary files (e.g., 'topol*' and 'posre*').
        """
        logger.info("Cleaning the PDB using GROMACS pdb2gmx...")
        if not in_pdb:
            in_pdb = self.inpdb
        self.gmx("pdb2gmx", f=in_pdb, o=in_pdb, **kwargs)
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def split_chains(self):
        """Splits the input PDB file into separate files for each chain.

        Nucleotide chains are saved to self.nucdir, while protein chains
        are saved to self.prodir. The determination is based on the
        residue names.
        """

        def it_is_nucleotide(atoms):
            # Check if the chain is nucleotide based on residue name.
            return atoms.resnames[0] in self.NUC_RESNAMES

        logger.info("Splitting chains from the input PDB...")
        system = pdbtools.pdb2system(self.inpdb)
        for chain in system.chains():
            atoms = chain.atoms
            if it_is_nucleotide(atoms):
                out_pdb = os.path.join(self.nucdir, f"chain_{chain.chid}.pdb")
            else:
                out_pdb = os.path.join(self.prodir, f"chain_{chain.chid}.pdb")
            atoms.write_pdb(out_pdb)

    def clean_chains_mm(self, **kwargs):
        """Cleans chain-specific PDB files using PDBfixer (OpenMM).

        Kwargs are passed to pdbtools.clean_pdb. Also renames chain IDs
        based on the file name.
        """
        kwargs.setdefault("add_missing_atoms", True)
        kwargs.setdefault("add_hydrogens", True)
        kwargs.setdefault("pH", 7.0)
        logger.info("Cleaning chain PDBs using OpenMM...")
        files = [os.path.join(self.prodir, f) for f in os.listdir(self.prodir)]
        files += [os.path.join(self.nucdir, f) for f in os.listdir(self.nucdir)]
        files = sorted(files)
        for file in files:
            pdbtools.clean_pdb(file, file, **kwargs)
            new_chain_id = file.split("chain_")[1][0]
            pdbtools.rename_chain_in_pdb(file, new_chain_id)

    def clean_chains_gmx(self, **kwargs):
        """Cleans chain-specific PDB files using GROMACS pdb2gmx tool.

        Args:
            kwargs: Additional keyword arguments for the GROMACS command.

        Processes all files in the protein and nucleotide directories, renaming chains
        and cleaning temporary files afterward.
        """
        logger.info("Cleaning chain PDBs using GROMACS pdb2gmx...")
        files = [
            os.path.join(self.prodir, f)
            for f in os.listdir(self.prodir)
            if not f.startswith("#")
        ]
        files += [
            os.path.join(self.nucdir, f)
            for f in os.listdir(self.nucdir)
            if not f.startswith("#")
        ]
        files = sorted(files)
        with cd(self.root):
            for file in files:
                new_chain_id = file.split("chain_")[1][0]
                self.gmx("pdb2gmx", f=file, o=file, **kwargs)
                pdbtools.rename_chain_and_histidines_in_pdb(file, new_chain_id)
            clean_dir(self.prodir)
            clean_dir(self.nucdir)
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def get_go_maps(self, append=False):
        """
        Retrieves GO contact maps for proteins using the RCSU server.
        http://info.ifpan.edu.pl/~rcsu/rcsu/index.html

        Args:
            append (bool, optional): If True, filters out maps that already exist in self.mapdir.
        """
        print("Getting GO-maps", file=sys.stderr)
        from reforge.martini import getgo

        pdbs = sorted(
            [os.path.join(self.prodir, file) for file in os.listdir(self.prodir)]
        )
        map_names = [f.replace("pdb", "map") for f in os.listdir(self.prodir)]
        if append:
            pdbs = [
                pdb
                for pdb, amap in zip(pdbs, map_names)
                if amap not in os.listdir(self.mapdir)
            ]
        if pdbs:
            getgo.get_go(self.mapdir, pdbs)
        else:
            print("Maps already there", file=sys.stderr)

    def martinize_proteins_go(self, append=False, **kwargs):
        """
        Performs virtual site-based GoMartini coarse-graining on protein PDBs.
        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize
        All **kwargs go directly to Martinize2.
        Run 'martinize2 -h' to see the full list of parameters

        Args:
            append (bool, optional): If True, only processes proteins for which
                                     corresponding topology files do not already exist.
            kwargs: Additional parameters for the martinize_go function.

        Generates .itp files and cleans temporary directories after processing.
        """
        logger.info("Working on proteins (GoMartini)...")
        from reforge.martini.martini_tools import martinize_go

        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace("pdb", "itp") for f in pdbs]
        if append:
            pdbs = [
                pdb
                for pdb, itp in zip(pdbs, itps)
                if itp not in os.listdir(self.topdir)
            ]
        else:
            clean_dir(self.topdir, "go_*.itp")
        # Create files for virtual CA parameters if they don't exist.
        file = os.path.join(self.topdir, "go_atomtypes.itp")
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("[ atomtypes ]\n")
        file = os.path.join(self.topdir, "go_nbparams.itp")
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("[ nonbond_params ]\n")
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            name = file.split(".")[0]
            go_map = os.path.join(self.mapdir, "{name}.map")
            martinize_go(self.root, self.topdir, in_pdb, cg_pdb, name=name, **kwargs)
        clean_dir(self.cgdir)
        clean_dir(self.root)
        clean_dir(self.root, "*.itp")

    def martinize_proteins_en(self, append=False, **kwargs):
        """
        Generates an elastic network for proteins using the Martini elastic network model.
        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize
        All **kwargs go directly to Martinize2.
        Run 'martinize2 -h' to see the full list of parameters

        Args:
            append (bool, optional): If True, processes only proteins that do not already
                                     have corresponding topology files.
            kwargs: Elastic network parameters (e.g., elastic bond force constants, cutoffs).

        Modifies the generated ITP files by replacing default molecule names with the actual
        protein names and cleans temporary files.
        """
        logger.info("Working on proteins (Elastic Network)...")
        from .martini.martini_tools import martinize_en

        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace("pdb", "itp") for f in pdbs]
        if append:
            pdbs = [
                pdb
                for pdb, itp in zip(pdbs, itps)
                if itp not in os.listdir(self.topdir)
            ]
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            new_itp = os.path.join(self.root, "molecule_0.itp")
            updated_itp = os.path.join(self.topdir, file.replace("pdb", "itp"))
            new_top = os.path.join(self.root, "protein.top")
            martinize_en(self.root, self.topdir, in_pdb, cg_pdb, **kwargs)
            # Replace 'molecule_0' with the actual molecule name in the ITP.
            with open(new_itp, "r", encoding="utf-8") as f:
                content = f.read()
            updated_content = content.replace("molecule_0", f"{file[:-4]}", 1)
            with open(updated_itp, "w", encoding="utf-8") as f:
                f.write(updated_content)
            os.remove(new_top)
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_nucleotides(self, **kwargs):
        """Performs coarse-graining on nucleotide PDBs using the
        martinize_nucleotide tool.

        Args:
            append (bool, optional): If True, skips already existing topologies.
            kwargs: Additional parameters for the martinize_nucleotide function.

        After processing, renames files and moves the resulting ITP files to the topology directory.
        """
        logger.info("Working on nucleotides...")
        from .martini.martini_tools import martinize_nucleotide

        for file in os.listdir(self.nucdir):
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            martinize_nucleotide(self.root, in_pdb, cg_pdb, **kwargs)
        nfiles = [f for f in os.listdir(self.root) if f.startswith("Nucleic")]
        for f in nfiles:
            file = os.path.join(self.root, f)
            command = f"sed -i s/Nucleic_/chain_/g {file}"
            sp.run(command.split())
            outfile = f.replace("Nucleic", "chain")
            shutil.move(
                os.path.join(self.root, file), os.path.join(self.topdir, outfile)
            )
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_rna(self, append=False, **kwargs):
        """Coarse-grains RNA molecules using the martinize_rna tool.

        Args:
            append (bool, optional): If True, processes only RNA files without existing topologies.
            kwargs: Additional parameters for the martinize_rna function.

        Exits the process with an error message if coarse-graining fails.
        """
        logger.info("Working on RNA molecules...")
        from reforge.martini.martini_tools import martinize_rna
        files = os.listdir(self.nucdir)
        if append:
            files = [f for f in files if f.replace('pdb', 'itp') not in self.topdir]
        for file in files:
            molname = file.split(".")[0]
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            cg_itp = os.path.join(self.topdir, molname + ".itp")
            try:
                martinize_rna(
                    self.root, f=in_pdb, os=cg_pdb, ot=cg_itp, mol=molname, **kwargs
                )
            except Exception as e:
                sys.exit(f"Could not coarse-grain {in_pdb}: {e}")

    def make_solute_pdb(self, **kwargs):
        """Merges coarse-grained PDB files into a single solute PDB file.

        Args:
            kwargs: Additional keyword arguments for the GROMACS editconf command. Defaults:
                - d: Distance parameter (default: 1.0)
                - bt: Box type (default: 'dodecahedron')

        Uses the AtomList from pdbtools to merge and renumber atoms, then calls the
        GROMACS 'editconf' command to finalize the solute PDB.
        """
        kwargs.setdefault("d", 1.0)
        kwargs.setdefault("bt", "dodecahedron")
        logger.info("Merging CG PDB files into a single solute PDB...")
        with cd(self.root):
            cg_pdb_files = os.listdir(self.cgdir)
            cg_pdb_files = sort_upper_lower_digit(cg_pdb_files)
            cg_pdb_files = [os.path.join(self.cgdir, fname) for fname in cg_pdb_files]
            all_atoms = AtomList()
            for file in cg_pdb_files:
                atoms = pdbtools.pdb2atomlist(file)
                all_atoms.extend(atoms)
            all_atoms.renumber()
            all_atoms.write_pdb(self.solupdb)
            self.gmx("editconf", f=self.solupdb, o=self.solupdb, **kwargs)

    def make_system_top(self, add_resolved_ions=False, prefix="chain"):
        """Creates the system topology file by including all relevant ITP files
        and defining the system and molecule sections.

        Args:
            add_resolved_ions (bool, optional): If True, counts and includes resolved ions.
            prefix (str, optional): Prefix for ITP files to include (default: 'chain').

        Writes the topology file (self.systop) with include directives and molecule counts.
        """
        logger.info("Writing system topology...")
        itp_files = [
            f
            for f in os.listdir(self.topdir)
            if f.startswith(prefix) and f.endswith(".itp")
        ]
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
                f.write('#include "topol/{filename}"\n')
            # System name and molecule count
            f.write("\n[ system ]\n")
            f.write("Martini system for {self.sysname}\n")
            f.write("\n[molecules]\n")
            f.write("; name\t\tnumber\n")
            for filename in itp_files:
                molecule_name = os.path.splitext(filename)[0]
                f.write("{molecule_name}\t\t1\n")
            # Add resolved ions if requested.
            if add_resolved_ions:
                ions = self.count_resolved_ions()
                for ion, count in ions.items():
                    if count > 0:
                        f.write(f"{ion}    {count}\n")

    def make_gro_file(self, d=1.25, bt="dodecahedron"):
        """Generates the final GROMACS GRO file from coarse-grained PDB files.

        Args:
            d (float, optional): Distance parameter for the editconf command (default: 1.25).
            bt (str, optional): Box type for the editconf command (default: 'dodecahedron').

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
            command = (
                f"gmx_mpi editconf -f {self.sysgro} -d {d} -bt {bt}  -o {self.sysgro}"
            )
            sp.run(command.split())

    def solvate(self, **kwargs):
        """Solvates the system using GROMACS solvate command.

        Args:
            kwargs: Additional parameters for the solvate command. Defaults:
                    - cp: 'solute.pdb'
                    - cs: 'water.gro'
        """
        kwargs.setdefault("cp", "solute.pdb")
        kwargs.setdefault("cs", "water.gro")
        self.gmx("solvate", p=self.systop, o=self.syspdb, **kwargs)

    def find_resolved_ions(self, mask=["MG", "ZN", "K"]):
        """Identifies resolved ions in the input PDB file and writes them to
        'ions.pdb'.

        Args:
            mask (list, optional): List of ion identifiers to look for (default: ['MG', 'ZN', 'K']).
        """
        mask_atoms(self.inpdb, "ions.pdb", mask=mask)

    def count_resolved_ions(self, ions=["MG", "ZN", "K"]):
        """Counts the number of resolved ions in the system PDB file.

        Args:
            ions (list, optional): List of ion names to count (default: ['MG', 'ZN', 'K']).

        Returns:
            dict: A dictionary mapping ion names to their counts.
        """
        counts = {ion: 0 for ion in ions}
        with open(self.syspdb, "r") as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    current_ion = line[12:16].strip()
                    if current_ion in ions:
                        counts[current_ion] += 1
        return counts

    def add_bulk_ions(self, solvent="W", **kwargs):
        """Adds bulk ions to neutralize the system using GROMACS genion.

        Args:
            solvent (str, optional): Solvent residue name (default: 'W').
            kwargs: Additional parameters for genion. Defaults include:
                    - conc: 0.15
                    - pname: 'NA'
                    - nname: 'CL'
                    - neutral: ''
        """
        kwargs.setdefault("conc", 0.15)
        kwargs.setdefault("pname", "NA")
        kwargs.setdefault("nname", "CL")
        kwargs.setdefault("neutral", "")
        self.gmx("grompp", f="mdp/ions.mdp", c=self.syspdb, p=self.systop, o="ions.tpr")
        self.gmx(
            "genion",
            clinput="W\n",
            s="ions.tpr",
            p=self.systop,
            o=self.syspdb,
            **kwargs,
        )
        self.gmx("editconf", f=self.syspdb, o=self.sysgro)
        clean_dir(self.root, "ions.tpr")

    def make_system_ndx(self, backbone_atoms=["CA", "P", "C1'"]):
        """Creates an index (NDX) file for the system, separating solute,
        backbone, solvent, and individual chains.

        Args:
            backbone_atoms (list, optional): List of atom names to include in the backbone (default: ["CA", "P", "C1'"]).
        """
        logger.info("Making index file...")
        system = pdbtools.pdb2atomlist(self.syspdb)
        solute = pdbtools.pdb2atomlist(self.solupdb)
        solvent = AtomList(system[len(solute) :])
        backbone = solute.mask(backbone_atoms, mode="name")
        system.write_ndx(self.sysndx, header=f"[ System ]", append=False, wrap=15)
        solute.write_ndx(self.sysndx, header=f"[ Solute ]", append=True, wrap=15)
        backbone.write_ndx(self.sysndx, header=f"[ Backbone ]", append=True, wrap=15)
        solvent.write_ndx(self.sysndx, header=f"[ Solvent ]", append=True, wrap=15)
        chids = sorted(set(solute.chids))
        for chid in chids:
            chain = solute.mask(chid, mode="chid")
            chain.write_ndx(
                self.sysndx, header=f"[ chain_{chid} ]", append=True, wrap=15
            )
        logger.info(f"Written index to {self.sysndx}")

    def get_mean_sem(self, pattern="dfi*.npy"):
        """Calculates the mean and standard error of the mean (SEM) from numpy
        files.

        Args:
            pattern (str, optional): Filename pattern to search for (default: 'dfi*.npy').

        Saves the calculated averages and errors as numpy files in the data directory.
        """
        logger.info(f"Calculating averages and errors from {pattern}")
        files = io.pull_files(self.mddir, pattern)
        datas = [np.load(file) for file in files]
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        file_mean = os.path.join(self.datdir, pattern.split("*")[0] + "_av.npy")
        file_err = os.path.join(self.datdir, pattern.split("*")[0] + "_err.npy")
        np.save(file_mean, mean)
        np.save(file_err, sem)

    def get_td_averages(self, fname, loop=True):
        """Calculates time-dependent averages from a set of numpy files.

        Args:
            fname (str): Filename pattern to pull files from the MD runs directory.
            loop (bool, optional): If True, processes files sequentially (default: True).

        Returns:
            numpy.ndarray: The time-dependent average.
        """
        logger.info("Getting time-dependent averages")
        files = io.pull_files(self.mddir, fname)
        if loop:
            logger.info(f"Processing {files[0]}")
            average = np.load(files[0])
            for f in files[1:]:
                logger.info(f"Processing {f}")
                arr = np.load(f)
                average += arr
            average /= len(files)
        else:
            arrays = [np.load(f) for f in files]
            average = np.average(arrays, axis=0)
        np.save(os.path.join(self.datdir, fname), average)
        logger.info("Done!")
        return average

    def get_averages(self, rmsf=False, dfi=True, dci=True):
        """Computes averages for various analyses (RMSF, DFI, DCI) based on MD
        run data.

        Args:
            rmsf (bool, optional): If True, computes RMSF averages.
            dfi (bool, optional): If True, computes DFI averages.
            dci (bool, optional): If True, computes DCI averages.
        """
        all_files = io.pull_all_files(self.mddir)
        if rmsf:
            files = io.filter_files(all_files, sw="rmsf.", ew=".xvg")
            self.get_mean_sem(files, f"rmsf.csv", col=1)
            for chain in self.chains:
                sw = f"rmsf_{chain}"
                files = io.filter_files(all_files, sw=sw, ew=".xvg")
                self.get_mean_sem(files, f"{sw}.csv", col=1)
        if dfi:
            print(f"Processing DFI", file=sys.stderr)
            files = io.filter_files(all_files, sw="dfi", ew=".xvg")
            self.get_mean_sem(files, f"dfi.csv", col=1)
        if dci:
            print(f"Processing DCI", file=sys.stderr)
            files = io.filter_files(all_files, sw="dci", ew=".xvg")
            self.get_mean_sem_2d(
                files, out_fname=f"dci.csv", out_errname=f"dci_err.csv"
            )
            files = io.filter_files(all_files, sw="asym", ew=".xvg")
            self.get_mean_sem_2d(
                files, out_fname=f"dci.csv", out_errname=f"dci_err.csv"
            )

    def initmd(self, runname):
        """Initializes a new MD run.

        Args:
            runname (str): Name for the MD run.

        Returns:
            MDRun: An instance of the MDRun class for the specified run.
        """
        mdrun = MDRun(self.sysdir, self.sysname, runname)
        return mdrun


class MDRun(GmxSystem):
    """Subclass of GmxSystem for executing molecular dynamics (MD) simulations
    and performing post-processing analyses."""

    def __init__(self, sysdir, sysname, runname):
        """Initializes the MD run environment with additional directories for
        analysis.

        Args:
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
        """Creates necessary directories for the MD run and copies essential
        files."""
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        shutil.copy("atommass.dat", os.path.join(self.rundir, "atommass.dat"))

    def empp(self, **kwargs):
        """Prepares the energy minimization run using GROMACS grompp.

        Args:
            kwargs: Additional parameters for grompp. Defaults include:
                    - f: Path to 'em.mdp' file.
                    - c: Structure file.
                    - r: Reference structure.
                    - p: Topology file.
                    - n: Index file.
                    - o: Output TPR file ('em.tpr').
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

        Args:
            kwargs: Additional parameters for grompp. Defaults include:
                    - f: Path to 'hu.mdp'.
                    - c: Starting structure ('em.gro').
                    - r: Reference structure ('em.gro').
                    - p: Topology file.
                    - n: Index file.
                    - o: Output TPR file ('hu.tpr').
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

        Args:
            kwargs: Additional parameters for grompp. Defaults include:
                    - f: Path to 'eq.mdp'.
                    - c: Starting structure ('hu.gro').
                    - r: Reference structure ('hu.gro').
                    - p: Topology file.
                    - n: Index file.
                    - o: Output TPR file ('eq.tpr').
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

        Args:
            grompp (bool, optional): Whether to run grompp (default: True).
            kwargs: Additional parameters for grompp. Defaults include:
                    - f: Path to 'md.mdp'.
                    - c: Starting structure ('eq.gro').
                    - r: Reference structure ('eq.gro').
                    - p: Topology file.
                    - n: Index file.
                    - o: Output TPR file ('md.tpr').
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

        Args:
            kwargs: Additional parameters for mdrun. Defaults include:
                    - deffnm: 'md'
                    - nsteps: '-2'
                    - ntomp: '8'
        """
        kwargs.setdefault("deffnm", "md")
        kwargs.setdefault("nsteps", "-2")
        kwargs.setdefault("ntomp", "8")
        with cd(self.rundir):
            cli.gmx_mdrun(**kwargs)

    def trjconv(self, clinput=None, **kwargs):
        """Converts trajectories using GROMACS trjconv.

        Args:
            clinput (str, optional): Input to be passed to trjconv.
            kwargs: Additional parameters for trjconv.
        """
        with cd(self.rundir):
            cli.gmx_trjconv(clinput=clinput, **kwargs)

    def rmsf(self, clinput=None, **kwargs):
        """Calculates RMSF using GROMACS rmsf.

        Args:
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

        Args:
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

        Args:
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

        Args:
            clinput (str, optional): Input for the clustering command.
            kwargs: Additional parameters for cluster.
        """
        kwargs.setdefault("s", self.str)
        kwargs.setdefault("f", self.trj)
        with cd(self.cludir):
            cli.gmx_cluster(clinput=clinput, **kwargs)

    def extract_cluster(self, clinput=None, **kwargs):
        """Extracts frames belonging to a cluster using GROMACS extract-
        cluster.

        Args:
            clinput (str, optional): Input for the extract-cluster command.
            kwargs: Additional parameters for extract-cluster. Defaults include:
                    - clusters: 'cluster.ndx'
        """
        kwargs.setdefault("f", self.trj)
        kwargs.setdefault("clusters", "cluster.ndx")
        with cd(self.cludir):
            cli.gmx_extract_cluster(clinput=clinput, **kwargs)

    def covar(self, clinput=None, **kwargs):
        """Calculates and diagonalizes the covariance matrix using GROMACS
        covar.

        Args:
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

        Args:
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
        """Prepares files for essential dynamics analysis using GROMACS make-
        edi.

        Args:
            clinput (str, optional): Input for the make-edi command.
            kwargs: Additional parameters for make-edi. Defaults include:
                    - f: Eigenvector file.
                    - s: Structure file.
        """
        kwargs.setdefault("f", "eigenvec.trr")
        kwargs.setdefault("s", "../traj.pdb")
        cli.gmx_make_edi(self.covdir, clinput=clinput, **kwargs)

    def get_covmats(
        self, u=None, ag=None, sample_rate=1, b=50000, e=1000000, n=10, outtag="covmat"
    ):
        """Calculates covariance matrices by splitting the trajectory into
        chunks.

        Args:
            u (MDAnalysis.Universe, optional): Pre-loaded MDAnalysis Universe; if None, creates one.
            ag (AtomGroup, optional): Atom selection; if None, selects backbone atoms.
            sample_rate (int, optional): Sampling rate for positions.
            b (int, optional): Begin time/frame.
            e (int, optional): End time/frame.
            n (int, optional): Number of covariance matrices to calculate.
            outtag (str, optional): Tag prefix for output files.
        """
        logger.info("Calculating covariance matrices...")
        if not u:
            u = mda.Universe(self.str, self.trj, in_memory=True)
        if not ag:
            ag = u.atoms.select_atoms("name BB or name BB1 or name BB3")
            if not ag:
                ag = u.atoms.select_atoms("name CA or name P or name C1'")
        positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e)
        mdm.calc_and_save_covmats(positions, outdir=self.covdir, n=n, outtag=outtag)
        logger.info("Finished calculating covariance matrices!")

    def get_pertmats(self, intag="covmat", outtag="pertmat", **kwargs):
        """Calculates perturbation matrices from the covariance matrices.

        Args:
            intag (str, optional): Input file tag for covariance matrices.
            outtag (str, optional): Output file tag for perturbation matrices.
            kwargs: Additional parameters for perturbation matrix calculation.
        """
        with cd(self.covdir):
            cov_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for cov_file in cov_files:
                logger.info(f"  Processing covariance matrix {cov_file}")
                covmat = np.load(cov_file)
                logger.info("  Calculating perturbation matrix")
                pertmat = mdm.perturbation_matrix(covmat)
                pert_file = cov_file.replace(intag, outtag)
                logger.info(f"  Saving perturbation matrix at {pert_file}")
                np.save(pert_file, pertmat)
        logger.info("Finished calculating perturbation matrices!")

    def get_dfi(self, intag="pertmat", outtag="dfi", **kwargs):
        """Calculates Dynamic Flexibility Index (DFI) from perturbation
        matrices.

        Args:
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DFI values.
            kwargs: Additional parameters for DFI calculation.
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info(f"  Processing perturbation matrix {pert_file}")
                pertmat = np.load(pert_file)
                logger.info("  Calculating DFI")
                dfi = mdm.dfi(pertmat)
                dfi_file = pert_file.replace(intag, outtag)
                dfi_file = os.path.join(self.covdir, dfi_file)
                np.save(dfi_file, dfi)
                logger.info(f"  Saved DFI at {dfi_file}")
        logger.info("Finished calculating DFIs!")

    def get_dci(self, intag="pertmat", outtag="dci", asym=False):
        """Calculates the Dynamic Coupling Index (DCI) from perturbation
        matrices.

        Args:
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DCI values.
            asym (bool, optional): If True, calculates asymmetric DCI.
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info(f"  Processing perturbation matrix {pert_file}")
                pertmat = np.load(pert_file)
                logger.info("  Calculating DCI")
                dci_file = pert_file.replace(intag, outtag)
                dci_file = os.path.join(self.covdir, dci_file)
                dci = mdm.dci(pertmat, asym=asym)
                np.save(dci_file, dci)
                logger.info(f"  Saved DCI at {dci_file}")
        logger.info("Finished calculating DCIs!")

    def get_group_dci(self, groups=[], labels=[], asym=False):
        """Calculates DCI between specified groups based on perturbation
        matrices.

        Args:
            groups (list): List of groups (atom indices or similar) to compare.
            labels (list): Corresponding labels for the groups.
            asym (bool, optional): If True, computes asymmetric group DCI.
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        logger.info(f"Working dir: {self.covdir}")
        pert_files = [f for f in sorted(os.listdir()) if f.startswith("pertmat")]
        for pert_file in pert_files:
            logger.info(f"  Processing perturbation matrix {pert_file}")
            pertmat = np.load(pert_file)
            logger.info("  Calculating group DCI")
            dcis = mdm.group_molecule_dci(pertmat, groups=groups, asym=asym)
            for dci, group, group_id in zip(dcis, groups, labels):
                dci_file = pert_file.replace("pertmat", f"gdci_{group_id}")
                dci_file = os.path.join(self.covdir, dci_file)
                np.save(dci_file, dci)
                logger.info(f"  Saved group DCI at {dci_file}")
            ch_dci_file = pert_file.replace("pertmat", f"ggdci")
            ch_dci_file = os.path.join(self.covdir, ch_dci_file)
            ch_dci = mdm.group_group_dci(pertmat, groups=groups, asym=asym)
            np.save(ch_dci_file, ch_dci)
            logger.info(f"  Saved group-group DCI at {ch_dci_file}")
        logger.info("Finished calculating group DCIs!")
        os.chdir(bdir)

    def get_rmsf_by_chain(self, **kwargs):
        """Calculates RMSF for each chain in the system using GROMACS rmsf.

        Args:
            kwargs: Additional parameters for the rmsf command. Defaults include:
                    - f: Trajectory file.
                    - s: Structure file.
                    - n: Index file.
                    - res: Whether to output per-residue RMSF (default: 'no').
                    - fit: Whether to fit the trajectory (default: 'yes').
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

        Args:
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


################################################################################
# Utils
################################################################################


def sort_upper_lower_digit(alist):
    """Sorts a list of strings such that uppercase letters come first, then
    lowercase letters, followed by digits. Useful for organizing GROMACS
    multichain files.

    Args:
        alist (iterable): List of strings to sort.

    Returns:
        list: Sorted list of strings.
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist
