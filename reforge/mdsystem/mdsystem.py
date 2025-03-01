"""File: mdsystem.py

Description:
    This module provides classes and functions for setting up, running, and
    analyzing molecular dynamics (MD) simulations.  The main
    classes include:

Usage:
  
Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - Pandas
    - The reforge package and its dependencies

Author: DY
Date: 2025-02-27
"""

import os
import sys
import shutil
import subprocess as sp
import numpy as np
from reforge import cli, mdm, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger
from reforge.martini import getgo, martini_tools

################################################################################
# GMX system class
################################################################################

class MDSystem:
    """
    Most attributes are paths to files and directories needed to set up
    and run the MD simulation.
    """
    NUC_RESNAMES = ["A", "C", "G", "U",
                    "RA3", "RA5", "RC3", "RC5", 
                    "RG3", "RG5", "RU3", "RU5",]

    def __init__(self, sysdir, sysname):
        """Initializes the MD system with required directories and file paths.

        Parameters
        ----------
            sysdir (str): Base directory for collection of MD systems
            sysname (str): Name of the MD system.

        Sets up paths for various files required for coarse-grained MD simulation.
        """
        self.sysname = sysname
        self.sysdir = os.path.abspath(sysdir)
        self.root = os.path.join(self.sysdir, sysname)
        self.inpdb = os.path.join(self.root, "inpdb.pdb")
        self.solupdb = os.path.join(self.root, "solute.pdb")
        self.syspdb = os.path.join(self.root, "system.pdb")
        self.prodir = os.path.join(self.root, "proteins")
        self.nucdir = os.path.join(self.root, "nucleotides")
        self.iondir = os.path.join(self.root, "ions")
        self.ionpdb = os.path.join(self.iondir, "ions.pdb")
        self.topdir = os.path.join(self.root, "topol")
        self.mapdir = os.path.join(self.root, "map")
        self.cgdir = os.path.join(self.root, "cgpdb")
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

    def sort_input_pdb(self, in_pdb="inpdb.pdb"):
        """Sorts and renames atoms and chains in the input PDB file.

        Parameters
        ----------
            in_pdb (str): Path to the input PDB file (default: "inpdb.pdb").

        Uses pdbtools to perform sorting and renaming, saving the result to self.inpdb.
        """
        with cd(self.root):
            pdbtools.sort_pdb(in_pdb, self.inpdb)

    def clean_pdb_mm(self, in_pdb=None, **kwargs):
        """Cleans the starting PDB file using PDBfixer (via OpenMM).

        Parameters
        ----------
            in_pdb (str, optional): Input PDB file to clean. If None, uses self.inpdb.
            kwargs: Additional keyword arguments for pdbtools.clean_pdb.
        """
        logger.info("Cleaning the PDB using OpenMM's PDBfixer...")
        if not in_pdb:
            in_pdb = self.inpdb
        pdbtools.clean_pdb(in_pdb, in_pdb, **kwargs)

    def clean_pdb_gmx(self, in_pdb=None, **kwargs):
        """Cleans the PDB file using GROMACS pdb2gmx tool.

        Parameters
        ----------
            in_pdb (str, optional): Input PDB file to clean. If None, uses self.inpdb.
            kwargs: Additional keyword arguments for the GROMACS command.

        After running pdb2gmx, cleans up temporary files (e.g., "topol*" and "posre*").
        """
        logger.info("Cleaning the PDB using GROMACS pdb2gmx...")
        if not in_pdb:
            in_pdb = self.inpdb
        with cd(self.root):
            cli.gmx("pdb2gmx", f=in_pdb, o=in_pdb, **kwargs)
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def split_chains(self):
        """Splits the input PDB file into separate files for each chain.

        Nucleotide chains are saved to self.nucdir, while protein chains are saved to self.prodir.
        The determination is based on the residue names.
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

        Kwargs are passed to pdbtools.clean_pdb. Also renames chain IDs based on the file name.
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

        Parameters
        ----------
            kwargs: Additional keyword arguments for the GROMACS command.

        Processes all files in the protein and nucleotide directories, renaming chains
        and cleaning temporary files afterward.
        """
        logger.info("Cleaning chain PDBs using GROMACS pdb2gmx...")
        files = [os.path.join(self.prodir, f) for f in os.listdir(self.prodir)\
            if not f.startswith("#")]
        files += [os.path.join(self.nucdir, f) for f in os.listdir(self.nucdir)\
            if not f.startswith("#")]
        files = sorted(files)
        with cd(self.root):
            for file in files:
                new_chain_id = file.split("chain_")[1][0]
                cli.gmx("pdb2gmx", f=file, o=file, **kwargs)
                pdbtools.rename_chain_and_histidines_in_pdb(file, new_chain_id)
            clean_dir(self.prodir)
            clean_dir(self.nucdir)
        clean_dir(self.root, "topol*")
        clean_dir(self.root, "posre*")

    def get_go_maps(self, append=False):
        """Retrieves GO contact maps for proteins using the RCSU server.
        
        http://info.ifpan.edu.pl/~rcsu/rcsu/index.html

        Parameters
        ----------
            append (bool, optional): If True, filters out maps that already exist in self.mapdir.
        """
        print("Getting GO-maps", file=sys.stderr)
        pdbs = sorted([os.path.join(self.prodir, file) for file in os.listdir(self.prodir)])
        map_names = [f.replace("pdb", "map") for f in os.listdir(self.prodir)]
        if append:
            pdbs = [pdb for pdb, amap in zip(pdbs, map_names)\
                if amap not in os.listdir(self.mapdir)]
        if pdbs:
            getgo.get_go(self.mapdir, pdbs)
        else:
            print("Maps already there", file=sys.stderr)

    def martinize_proteins_go(self, append=False, **kwargs):
        """Performs virtual site-based GoMartini coarse-graining on protein PDBs.

        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize.
        All keyword arguments are passed directly to Martinize2. 
        Run `martinize2 -h` to see the full list of parameters.

        Parameters
        ----------
            append (bool, optional): If True, only processes proteins for 
                which corresponding topology files do not already exist.
            kwargs: Additional parameters for the martinize_go function.

        Generates .itp files and cleans temporary directories after processing.
        """
        logger.info("Working on proteins (GoMartini)...")
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace("pdb", "itp") for f in pdbs]
        if append:
            pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        else:
            clean_dir(self.topdir, "go_*.itp")
        # Create files for virtual CA parameters if they don't exist.
        file_path = os.path.join(self.topdir, "go_atomtypes.itp")
        if not os.path.isfile(file_path):
            with open(file_path, "w", encoding='utf-8') as f:
                f.write("[ atomtypes ]\n")
        file_path = os.path.join(self.topdir, "go_nbparams.itp")
        if not os.path.isfile(file_path):
            with open(file_path, "w", encoding='utf-8') as f:
                f.write("[ nonbond_params ]\n")
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            name = file.split(".")[0]
            # Note: Use f-string formatting correctly.
            go_map = os.path.join(self.mapdir, f"{name}.map")
            martini_tools.martinize_go(self.root, self.topdir, in_pdb, cg_pdb, name=name, **kwargs)
        clean_dir(self.cgdir)
        clean_dir(self.root)
        clean_dir(self.root, "*.itp")

    def martinize_proteins_en(self, append=False, **kwargs):
        """Generates an elastic network for proteins using the Martini elastic network model.

        Uses Martinize2 from https://github.com/marrink-lab/vermouth-martinize.
        All keyword arguments are passed directly to Martinize2. 
        Run `martinize2 -h` to see the full list of parameters.

        Parameters
        ----------
            append (bool, optional): If True, processes only proteins that do not 
                already have corresponding topology files.
            kwargs: Elastic network parameters (e.g., elastic bond force constants, cutoffs).

        Modifies the generated ITP files by replacing the default molecule name 
        with the actual protein name and cleans temporary files.
        """
        logger.info("Working on proteins (Elastic Network)...")
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace("pdb", "itp") for f in pdbs]
        if append:
            pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            new_itp = os.path.join(self.root, "molecule_0.itp")
            updated_itp = os.path.join(self.topdir, file.replace("pdb", "itp"))
            new_top = os.path.join(self.root, "protein.top")
            martini_tools.martinize_en(self.root, self.topdir, in_pdb, cg_pdb, **kwargs)
            # Replace 'molecule_0' with the actual molecule name in the ITP.
            with open(new_itp, "r", encoding="utf-8") as f:
                content = f.read()
            updated_content = content.replace("molecule_0", file[:-4], 1)
            with open(updated_itp, "w", encoding="utf-8") as f:
                f.write(updated_content)
            os.remove(new_top)
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_nucleotides(self, **kwargs):
        """Performs coarse-graining on nucleotide PDBs using the martinize_nucleotide tool.

        Parameters
        ----------
            append (bool, optional): If True, skips already existing topologies.
            kwargs: Additional parameters for the martinize_nucleotide function.

        After processing, renames files and moves the resulting ITP files to the topology directory.
        """
        logger.info("Working on nucleotides...")
        for file in os.listdir(self.nucdir):
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            martini_tools.martinize_nucleotide(self.root, in_pdb, cg_pdb, **kwargs)
        nfiles = [f for f in os.listdir(self.root) if f.startswith("Nucleic")]
        for f in nfiles:
            file_path = os.path.join(self.root, f)
            command = f"sed -i s/Nucleic_/chain_/g {file_path}"
            sp.run(command.split(), check=True)
            outfile = f.replace("Nucleic", "chain")
            shutil.move(file_path, os.path.join(self.topdir, outfile))
        clean_dir(self.cgdir)
        clean_dir(self.root)

    def martinize_rna(self, append=False, **kwargs):
        """Coarse-grains RNA molecules using the martinize_rna tool.

        Parameters
        ----------
            append (bool, optional): If True, processes only RNA files without existing topologies.
            kwargs: Additional parameters for the martinize_rna function.

        Exits the process with an error message if coarse-graining fails.
        """
        logger.info("Working on RNA molecules...")
        files = os.listdir(self.nucdir)
        if append:
            files = [f for f in files if f.replace("pdb", "itp") not in self.topdir]
        for file in files:
            molname = file.split(".")[0]
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            cg_itp = os.path.join(self.topdir, molname + ".itp")
            try:
                martini_tools.martinize_rna(self.root,
                    f=in_pdb, os=cg_pdb, ot=cg_itp, mol=molname, **kwargs)
            except Exception as e:
                sys.exit(f"Could not coarse-grain {in_pdb}: {e}")

    def make_cg_solute_pdb(self, **kwargs):
        """Merges coarse-grained PDB files into a single solute PDB file.

        Parameters
        ----------
            kwargs: Additional keyword arguments for the GROMACS editconf command. Defaults:
                - d: Distance parameter (default: 1.0)
                - bt: Box type (default: "dodecahedron").

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
            cli.gmx("editconf", f=self.solupdb, o=self.solupdb, **kwargs)

    def find_resolved_ions(self, mask=["MG", "ZN", "K"]):
        """Identifies resolved ions in the input PDB file and writes them to "ions.pdb".

        Parameters
        ----------
            mask (list, optional): List of ion identifiers to look for (default: ["MG", "ZN", "K"]).
        """
        pdbtools.mask_atoms(self.inpdb, "ions.pdb", mask=mask)

    def count_resolved_ions(self, ions=["MG", "ZN", "K"]):
        """Counts the number of resolved ions in the system PDB file.

        Parameters
        ----------
        ions (list, optional): 
            List of ion names to count (default: ["MG", "ZN", "K"]).

        Returns
        -------  
        dict: 
            A dictionary mapping ion names to their counts.
        """
        counts = {ion: 0 for ion in ions}
        with open(self.syspdb, "r", encoding='utf-8') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    current_ion = line[12:16].strip()
                    if current_ion in ions:
                        counts[current_ion] += 1
        return counts

    def get_mean_sem(self, pattern="dfi*.npy"):
        """Calculates the mean and standard error of the mean (SEM) from numpy files.

        Parameters
        ----------
            pattern (str, optional): Filename pattern to search for (default: "dfi*.npy").

        Saves the calculated averages and errors as numpy files in the data directory.
        """
        logger.info("Calculating averages and errors from %s", pattern)
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

        Parameters
        ----------
            fname (str): Filename pattern to pull files from the MD runs directory.
            loop (bool, optional): If True, processes files sequentially (default: True).

        Returns:
            numpy.ndarray: The time-dependent average.
        """
        logger.info("Getting time-dependent averages")
        files = io.pull_files(self.mddir, fname)
        if loop:
            logger.info("Processing %s", files[0])
            average = np.load(files[0])
            for f in files[1:]:
                logger.info("Processing %s", f)
                arr = np.load(f)
                average += arr
            average /= len(files)
        else:
            arrays = [np.load(f) for f in files]
            average = np.average(arrays, axis=0)
        np.save(os.path.join(self.datdir, fname), average)
        logger.info("Done!")
        return average


class MDRun(MDSystem):
    """Subclass of MDSystem for executing molecular dynamics (MD) simulations
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
    
    def prepare_files(self):
        """Creates necessary directories for the MD run and copies essential files."""
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        
    def get_covmats(self, u, ag, **kwargs):
        """Calculates covariance matrices by splitting the trajectory into chunks.

        Parameters
        ----------
            u (MDAnalysis.Universe, optional): Pre-loaded MDAnalysis Universe; if None, creates one.
            ag (AtomGroup, optional): Atom selection; if None, selects backbone atoms.
            sample_rate (int, optional): Sampling rate for positions.
            b (int, optional): Begin time/frame.
            e (int, optional): End time/frame.
            n (int, optional): Number of covariance matrices to calculate.
            outtag (str, optional): Tag prefix for output files.
        """
        sample_rate = kwargs.pop('sample_rate', 1)
        b = kwargs.pop('b', 50000)
        e = kwargs.pop('e', 1000000)
        n = kwargs.pop('b', 10)
        outtag = kwargs.pop('outtag', 'covmat')
        logger.info("Calculating covariance matrices...")
        positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e)
        mdm.calc_and_save_covmats(positions, outdir=self.covdir, n=n, outtag=outtag)
        logger.info("Finished calculating covariance matrices!")

    def get_pertmats(self, intag="covmat", outtag="pertmat"):
        """Calculates perturbation matrices from the covariance matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for covariance matrices.
            outtag (str, optional): Output file tag for perturbation matrices.
            kwargs: Additional parameters for perturbation matrix calculation.
        """
        with cd(self.covdir):
            cov_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for cov_file in cov_files:
                logger.info("  Processing covariance matrix %s", cov_file)
                covmat = np.load(cov_file)
                logger.info("  Calculating perturbation matrix")
                pertmat = mdm.perturbation_matrix(covmat)
                pert_file = cov_file.replace(intag, outtag)
                logger.info("  Saving perturbation matrix at %s", pert_file)
                np.save(pert_file, pertmat)
        logger.info("Finished calculating perturbation matrices!")

    def get_dfi(self, intag="pertmat", outtag="dfi"):
        """Calculates Dynamic Flexibility Index (DFI) from perturbation matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DFI values.
            kwargs: Additional parameters for DFI calculation.
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info("  Processing perturbation matrix %s", pert_file)
                pertmat = np.load(pert_file)
                logger.info("  Calculating DFI")
                dfi_val = mdm.dfi(pertmat)
                dfi_file = pert_file.replace(intag, outtag)
                dfi_file = os.path.join(self.covdir, dfi_file)
                np.save(dfi_file, dfi_val)
                logger.info("  Saved DFI at %s", dfi_file)
        logger.info("Finished calculating DFIs!")

    def get_dci(self, intag="pertmat", outtag="dci", asym=False):
        """Calculates the Dynamic Coupling Index (DCI) from perturbation matrices.

        Parameters
        ----------
            intag (str, optional): Input file tag for perturbation matrices.
            outtag (str, optional): Output file tag for DCI values.
            asym (bool, optional): If True, calculates asymmetric DCI.
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info("  Processing perturbation matrix %s", pert_file) 
                pertmat = np.load(pert_file)
                logger.info("  Calculating DCI")
                dci_file = pert_file.replace(intag, outtag)
                dci_file = os.path.join(self.covdir, dci_file)
                dci_val = mdm.dci(pertmat, asym=asym)
                np.save(dci_file, dci_val)
                logger.info("  Saved DCI at %s", dci_file)
        logger.info("Finished calculating DCIs!")

    def get_group_dci(self, groups, labels, asym=False):
        """Calculates DCI between specified groups based on perturbation matrices.

        Parameters
        ----------
            groups (list): List of groups (atom indices or similar) to compare.
            labels (list): Corresponding labels for the groups.
            asym (bool, optional): If True, computes asymmetric group DCI.
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        logger.info("Working dir: %s", self.covdir)
        pert_files = [f for f in sorted(os.listdir()) if f.startswith("pertmat")]
        for pert_file in pert_files:
            logger.info("  Processing perturbation matrix %s", pert_file) 
            pertmat = np.load(pert_file)
            logger.info("  Calculating group DCI")
            dcis = mdm.group_molecule_dci(pertmat, groups=groups, asym=asym)
            for dci_val, group, group_id in zip(dcis, groups, labels):
                dci_file = pert_file.replace("pertmat", f"gdci_{group_id}")
                dci_file = os.path.join(self.covdir, dci_file)
                np.save(dci_file, dci_val)
                logger.info("  Saved group DCI at %s", dci_file)
            ch_dci_file = pert_file.replace("pertmat", "ggdci")
            ch_dci_file = os.path.join(self.covdir, ch_dci_file)
            ch_dci = mdm.group_group_dci(pertmat, groups=groups, asym=asym)
            np.save(ch_dci_file, ch_dci)
            logger.info("  Saved group-group DCI at %s", ch_dci_file)
        logger.info("Finished calculating group DCIs!")
        os.chdir(bdir)

################################################################################
# Utils
################################################################################

def sort_upper_lower_digit(alist):
    """Sorts a list of strings such that uppercase letters come first, 
    then lowercase letters, followed by digits.

    This is useful for organizing GROMACS multichain files.

    Parameters
        ----------
        alist (iterable): List of strings to sort.

    Returns:
        list: Sorted list of strings.
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist
