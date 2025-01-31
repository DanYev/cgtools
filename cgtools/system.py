import os
import sys
import MDAnalysis as mda
import numpy as np
import pandas as pd
import shutil
import subprocess as sp
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO, Atom
from . import cli
from . import dci_dfi as dd


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

class CGSystem:
    """
    Class to set up and analyze protein-nucliotide-lipid systems for MD with GROMACS
    All the attributes are the paths to files and directories needed to set up and run CG MD
    """    
    DATADIR = 'cgtools/data'
    ITPDIR = 'cgtools/itp'
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
        self.sysgro     = os.path.join(self.wdir, 'system.gro')
        self.syspdb     = os.path.join(self.wdir, 'system.pdb')
        self.systop     = os.path.join(self.wdir, 'system.top')
        self.sysndx     = os.path.join(self.wdir, 'system.ndx')
        self.mdcpdb     = os.path.join(self.wdir, 'mdc.pdb')
        self.mdcndx     = os.path.join(self.wdir, 'mdc.ndx')
        self.bbndx      = os.path.join(self.wdir, 'bb.ndx')
        self.trjpdb     = os.path.join(self.wdir, 'traj.pdb')
        self.trjndx     = os.path.join(self.wdir, 'traj.ndx')
        self.prodir     = os.path.join(self.wdir, 'proteins')
        self.nucdir     = os.path.join(self.wdir, 'nucleotides')
        self.iondir     = os.path.join(self.wdir, 'ions')
        self.ionpdb     = os.path.join(self.iondir, 'ions.pdb')
        self.topdir     = os.path.join(self.wdir, 'topol')
        self.mapdir     = os.path.join(self.wdir, 'map')
        self.mdpdir     = os.path.join(self.wdir, 'mdp')
        self.cgdir      = os.path.join(self.wdir, 'cgpdb')
        self.grodir     = os.path.join(self.wdir, 'gro')
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
        for file in os.listdir(self.DATADIR):
            if file.endswith('.mdp'):
                fpath = os.path.join(self.DATADIR, file)
                outpath = os.path.join(self.mdpdir, file)
                shutil.copy(fpath, outpath)
        shutil.copy(os.path.join(self.DATADIR, 'water.gro'), self.wdir)
        for file in os.listdir(self.ITPDIR):
            if file.endswith('.itp'):
                fpath = os.path.join(self.ITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                shutil.copy(fpath, outpath)
                
    def make_ref_pdb(self):
        os.chdir(self.wdir)
        def rearrange_chains_and_renumber_atoms(input_pdb, output_pdb):
            """
            Rearrange chains in a PDB file alphabetically by chain ID and renumber atom IDs.
        
            Parameters:
                input_pdb (str): Path to the input PDB file.
                output_pdb (str): Path to save the rearranged PDB file.
            """
            with open(input_pdb, 'r') as file:
                pdb_lines = file.readlines()
        
            # Group lines by chain ID
            chain_dict = {}
            other_lines = []
        
            for line in pdb_lines:
                if line.startswith(("ATOM", )):
                    chain_id = line[21]  # Extract chain ID (column 22, index 21)
                    if chain_id not in chain_dict:
                        chain_dict[chain_id] = []
                    chain_dict[chain_id].append(line)
                else:
                    # Keep non-ATOM, HETATM, and TER lines (e.g., HEADER, REMARK, END)
                    other_lines.append(line)
        
            # Sort chains alphabetically
            sorted_chain_ids = sort_uld(chain_dict.keys())
        
            # Renumber atom IDs and write to the output file
            atom_id = 1  # Start atom ID renumbering
            with open(output_pdb, 'w') as file:
                # Write other (non-ATOM) lines first
                for line in other_lines:
                    file.write(line)
        
                # Write the sorted chains with updated atom IDs
                for chain_id in sorted_chain_ids:
                    for line in chain_dict[chain_id]:
                        if line.startswith(("ATOM", )):
                            # Update the atom ID (columns 7-11, index 6-11)
                            updated_line = f"{line[:6]}{atom_id:5d}{line[11:]}"
                            file.write(updated_line)
                            atom_id += 1  # Increment atom ID
                            if atom_id > 99999:
                                atom_id = 1
                        else:
                            # Write TER lines as-is
                            file.write(line)

        
        # Example usage
        input_pdb = "rearranged.pdb"  # Replace with your input file path
        output_pdb = "ref.pdb"  # Replace with your desired output file path
        rearrange_chains_and_renumber_atoms(input_pdb, output_pdb)
                
        
    def clean_inpdb(self, **kwargs):
        """
        Cleans starting PDB file using PDBfixer by OpenMM
        """
        print("Cleaning the PDB", file=sys.stderr)
        from pdbtools import prepare_aa_pdb
        in_pdb = self.inpdb
        out_pdb = in_pdb.replace('.pdb', '_clean.pdb')
        prepare_aa_pdb(in_pdb, out_pdb, **kwargs)  
        
    def clean_proteins(self, **kwargs):
        """
        Cleans protein PDB files using PDBfixer by OpenMM
        """
        print("Cleaning protein PDBs", file=sys.stderr)
        from pdbtools import prepare_aa_pdb, rename_chain
        files = [os.path.join(self.prodir, f) for f in os.listdir(self.prodir) if not f.endswith('_clean.pdb')]
        files = sorted(files)
        for in_pdb in files:
            print(f"Processing {in_pdb}", file=sys.stderr)
            out_pdb = in_pdb.replace('.pdb', '_clean.pdb')
            prepare_aa_pdb(in_pdb, out_pdb, **kwargs)  
            os.remove(in_pdb)
            old_chain_id = 'A'
            new_chain_id = in_pdb.split('chain_')[-1][0]
            rename_chain(out_pdb, in_pdb, old_chain_id, new_chain_id)
            os.remove(out_pdb)
    
    def split_chains(self, from_clean=False):
        """
        Cleans a separate PDB file for each chain in the initial structure
        """
        parser = PDBParser()
        in_pdb = self.inpdb
        if from_clean:
            in_pdb = self.inpdb.replace('.pdb', '_clean.pdb')
        structure = parser.get_structure(self.sysname, in_pdb)
        io = PDBIO()
        for model in structure:
            for chain in model:
                io.set_structure(chain)
                chain_id = chain.id
                if chain.get_unpacked_list()[0].get_resname() in self.NUC_RESNAMES:
                    out_pdb = os.path.join(self.nucdir, f'chain_{chain_id}.pdb')
                else:
                    out_pdb = os.path.join(self.prodir, f'chain_{chain_id}.pdb')
                io.save(out_pdb)
                
    def get_go_maps(self):
        """
        Get go contact maps for proteins using RCSU server
        """
        print('Getting GO-maps', file=sys.stderr)
        from get_go import get_go
        pdbs = [os.path.join(self.prodir, file) for file in os.listdir(self.prodir)]
        map_names = [f.replace('pdb', 'map') for f in os.listdir(self.prodir)]
        # Filter out existing maps
        pdbs = [pdb for pdb, amap in zip(pdbs, map_names) if amap not in os.listdir(self.mapdir)]
        if pdbs:
            get_go(self.mapdir, pdbs)
        else:
            print('Maps already there', file=sys.stderr)
        
    def martinize_proteins_go(self, **kwargs):
        """
        Virtual site based GoMartini:
        -go_map         Contact map to be used for the Martini Go model.Currently, only one format is supported. (default: None)
        -go_moltype     Set the name of the molecule when using Virtual Sites GoMartini. (default: protein)
        -go_eps         The strength of the Go model structural bias in kJ/mol. (default: 9.414)                        
        -go_low         Minimum distance (nm) below which contacts are removed. (default: 0.3)
        -go_up          Maximum distance (nm) above which contacts are removed. (default: 1.1)
        -go_res_dist    Minimum graph distance (similar sequence distance) below which contacts are removed. (default: 3)
        -resid          How to handle resid. Choice of mol or input. mol: resids are numbered from 1 to n for each molecule input: 
                        resids are the same as in the input pdb (default: mol)
        """
        print("Working on proteins", file=sys.stderr)
        # Make itp files to dump all the virtual CA's parameters into
        file = os.path.join(self.topdir, 'go_atomtypes.itp')
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                f.write(f'[ atomtypes ]\n')
        file = os.path.join(self.topdir, 'go_nbparams.itp')
        if not os.path.isfile(file):
            with open(file, 'w') as f:        
                f.write(f'[ nonbond_params ]\n')
        # Actually martinizing
        from martini_tools import martinize_go
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace('pdb', 'itp') for f in pdbs]
        # Filter out existing topologies
        pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            go_moltype = file.split('.')[0]
            go_map = os.path.join(self.mapdir, f'{go_moltype}.map')
            martinize_go(self.wdir, self.topdir, in_pdb, cg_pdb, go_moltype=go_moltype, go=go_map, **kwargs)
 
    def martinize_proteins_en(self, **kwargs):
        """
        Protein elastic network:
          -elastic              Write elastic bonds (default: False)
          -ef RB_FORCE_CONSTANT
                                Elastic bond force constant Fc in kJ/mol/nm^2 (default: 500)
          -el RB_LOWER_BOUND    Elastic bond lower cutoff: F = Fc if rij < lo (default: 0)
          -eu RB_UPPER_BOUND    Elastic bond upper cutoff: F = 0 if rij > up (default: 0.9)
          -ermd RES_MIN_DIST    The minimum separation between two residues to have an RB the default value is set by the force-field. (default: None)
          -ea RB_DECAY_FACTOR   Elastic bond decay factor a (default: 0)
          -ep RB_DECAY_POWER    Elastic bond decay power p (default: 1)
          -em RB_MINIMUM_FORCE  Remove elastic bonds with force constant lower than this (default: 0)
          -eb RB_SELECTION      Comma separated list of bead names for elastic bonds (default: None)
          -eunit RB_UNIT        Establish what is the structural unit for the elastic network. 
                                Bonds are only created within a unit. Options are molecule, chain, all, or aspecified region defined by resids,
                                with followingformat: <start_resid_1>:<end_resid_1>, <start_resid_2>:<end_resid_2>... (default: molecule)
        """
        print("Working on proteins", file=sys.stderr)
        # Actually martinizing
        from martini_tools import martinize_en
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace('pdb', 'itp') for f in pdbs]
        # Filter out existing topologies
        pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            new_itp = os.path.join(self.wdir, 'molecule_0.itp')
            new_top = os.path.join(self.wdir, 'protein.top')
            martinize_en(self.wdir, in_pdb, cg_pdb,  **kwargs) 
            shutil.move(new_itp, os.path.join(self.topdir, file.replace('pdb', 'itp')))
            os.remove(new_top)
    
    def martinize_nucleotides(self, **kwargs):
        print("Working on nucleotides", file=sys.stderr)
        from martini_tools import martinize_nucleotide
        for file in os.listdir(self.nucdir):
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            martinize_nucleotide(self.wdir, in_pdb, cg_pdb, **kwargs)
        bdir = os.getcwd()
        nfiles = [f for f in os.listdir(self.wdir) if f.startswith('Nucleic')]
        for f in nfiles:
            file = os.path.join(self.wdir, f)
            command = f'sed -i s/Nucleic_/chain_/g {file}'
            sp.run(command.split())
            outfile = f.replace('Nucleic', 'chain')
            shutil.move(os.path.join(self.wdir, file), os.path.join(self.topdir, outfile))

    def make_cgpdb_file(self, add_ions=False, **kwargs):
        with open(self.syspdb, 'w') as outfile:
            pass  # makes a clean new file
        with open(self.syspdb, 'a') as outfile:
            for filename in sorted(os.listdir(self.cgdir)):
                with open(os.path.join(self.cgdir, filename), 'r') as infile:
                    outfile.writelines(line for line in infile if line.startswith('ATOM'))
            if add_ions:
                with open(self.ionpdb, 'r') as infile:
                    for line in infile:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            line = line[:21] + ' ' + line[22:] # dont need the chain ID
                            outfile.write(line)
        cli.gmx_editconf(self.wdir, **kwargs)
        
    def make_topology_file(self, ions=['K','MG',]):
        itp_files = sorted([f for f in os.listdir(self.topdir) if f.startswith('chain')]) #
        ions = self.count_resolved_ions(ions=ions) 
        with open(self.systop, 'w') as f:
            # Include section
            f.write(f'#define GO_VIRT"\n')
            f.write(f'#define RUBBER_BANDS\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_rna.itp"\n')
            f.write(f'#include "{self.topdir}/martini_ions.itp"\n')
            if 'go_atomtypes.itp' in os.listdir(self.topdir):
                f.write(f'#include "{self.topdir}/go_atomtypes.itp"\n')
                f.write(f'#include "{self.topdir}/go_nbparams.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_solvents_v1.itp"\n') 
            f.write(f'#include "{self.topdir}/martini_v3.0.0_phospholipids_v1.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_ions_v1.itp"\n')
            f.write(f'\n')
            for filename in itp_files:
                if filename.endswith('.itp'):
                    filepath = os.path.join(self.topdir, filename)
                    f.write(f'#include "{filepath}"\n')
            # System name
            f.write(f'\n[ system ]\n')
            f.write(f'Martini system for {self.sysname}\n') 
            # Molecules
            f.write('\n[molecules]\n')
            f.write('; name\t\tnumber\n')
            for filename in itp_files:
                if filename.endswith('.itp'):
                    molecule_name = os.path.splitext(filename)[0]
                    f.write(f'{molecule_name}\t\t1\n')
            # Ions
            for ion, count in ions.items():
                if count > 0:
                    f.write(f'{ion}    {count}\n')
    
    @staticmethod                
    def count_itp_atoms(file_path):
        in_atoms_section = False
        atom_count = 0
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Strip whitespace and check if it's a comment or empty line
                    line = line.strip()
                    if not line or line.startswith(';'):
                        continue
                    # Detect the start of the [ atoms ] section
                    if line.startswith("[ atoms ]"):
                        in_atoms_section = True
                        continue
                    # Detect the start of a new section
                    if in_atoms_section and line.startswith('['):
                        break
                    # Count valid lines in the [ atoms ] section
                    if in_atoms_section:
                        atom_count += 1
            return atom_count
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return 0
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0
           
    def make_gro_file(self, d=1.25, bt='dodecahedron'):
        cg_pdb_files = sorted(os.listdir(self.cgdir))
        for file in cg_pdb_files:
            if file.endswith('.pdb'):
                pdb_file = os.path.join(self.cgdir, file)
                gro_file = pdb_file.replace('.pdb', '.gro').replace('cgpdb', 'gro')
                command = f'gmx_mpi editconf -f {pdb_file} -o {gro_file}'
                sp.run(command.split())
        # Merge all .gro files
        gro_files = sorted(os.listdir(self.grodir))
        total_count = 0
        for filename in gro_files:
            if filename.endswith(".gro"):
                filepath = os.path.join(self.grodir, filename)
                with open(filepath, 'r') as in_f:
                    atom_count = int(in_f.readlines()[1].strip())
                    total_count += atom_count
        with open(self.sysgro, 'w') as out_f:
            out_f.write(f"{self.sysname} \n")
            out_f.write(f"  {total_count}\n")
            for filename in gro_files:
                if filename.endswith(".gro"):
                    filepath = os.path.join(self.grodir, filename)
                    with open(filepath, 'r') as in_f:
                        lines = in_f.readlines()[2:-1]
                        for line in lines:
                            out_f.write(line)
            out_f.write("10.00000   10.00000   10.00000\n")  
        command = f'gmx_mpi editconf -f {self.sysgro} -d {d} -bt {bt}  -o {self.sysgro}'
        sp.run(command.split())
        
    def solvate(self, **kwargs):
        cli.gmx_solvate(self.wdir, **kwargs)
        
    def find_resolved_ions(self):
        mask_atoms(self.inpdb, 'ions.pdb', mask=['MG', 'ZN', 'K'])
        
    def count_resolved_ions(self, ions=['K','MG',]): # does NOT work for CA atoms for now
        counts = {ion: 0 for ion in ions}
        with open(self.syspdb, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    current_ion = line[12:16].strip()
                    if current_ion in ions:
                        counts[current_ion] += 1
        return counts
        
    def add_ions(self, conc=0.15, pname='NA', nname='CL'): 
        bdir = os.getcwd()
        os.chdir(self.wdir)
        command = f'gmx_mpi grompp -f mdp/ions.mdp -c {self.syspdb} -p system.top -o ions.tpr'
        sp.run(command.split())
        command = f'gmx_mpi genion -s ions.tpr -p system.top -conc {conc} -neutral -pname {pname} -nname {nname} -o {self.syspdb}'
        sp.run(command.split(), input='W\n', text=True) 
        os.chdir(bdir)
        cli.gmx_editconf(self.wdir, f=self.syspdb, o=self.sysgro)
        
    def initmd(self, runname):
        mdrun = MDRun(runname, self.sysdir, self.sysname)
        # self._mdruns.append(mdrun.runname)
        return mdrun

    def make_ndx(self, pdb, ndx='index.ndx', groups=[[]], **kwargs):
        cli.gmx_make_ndx(self.wdir, clinput='keep 0\n q\n', f=pdb, o=ndx, **kwargs)
        for atoms in groups:
            ndxstr = 'a ' + ' | a '.join(atoms) + '\n q \n'
            instr_a = '_'.join(atoms) + '\n'
            cli.gmx_make_ndx(self.wdir, clinput=ndxstr, f=pdb, o=ndx, n=ndx, **kwargs)
  
    @staticmethod    
    def mask_pdb(inpdb, outpdb, mask=['BB', 'BB1', 'BB2']):
        """
        Makes a masked PDB file from the input PDB file
        Args:
            inpdb(str): full path to the input pdb
            outpdb(str): full path to the output pdb
        """
        filtered_atoms = []
        with open(inpdb, 'r') as file:
            for line in file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    current_atom_name = line[12:16].strip()
                    if current_atom_name in mask:
                        # Copy chain ID to segID
                        chain_id = line[21].strip()  # Extract chain ID
                        seg_id = f"{chain_id:<4}"    # Format segID to occupy 4 characters
                        updated_line = line[:72] + seg_id + line[76:]  # Update segID field
                        filtered_atoms.append(updated_line)
        with open(outpdb, 'w') as output_file:
            for i, atom_line in enumerate(filtered_atoms, start=1):
                new_serial_number = f'{i:>5}'  
                new_atom_line = atom_line[:6] + new_serial_number + atom_line[11:]
                output_file.write(new_atom_line)
                
    def make_mdc_pdb_ndx(self, mask=['BB', 'BB1', 'BB2'], **kwargs):
        """
        Makes a pdb file of masked selected atoms and a ndx file with separated chains for this pdb
        Default is to select backbone atoms for proteins and RNA
        """
        CGSystem.mask_pdb(inpdb=self.syspdb, outpdb=self.mdcpdb, mask=mask)
        CGSystem.make_index_by_chain(self.wdir, inpdb=self.mdcpdb, chains=self.chains, outndx=self.mdcndx)
        
    def make_trj_pdb_ndx(self, mask=['BB', 'BB2', ], **kwargs):
        """
        Makes a pdb file of masked selected atoms and a ndx file with separated chains for this pdb
        Default is to select backbone atoms for proteins and RNA
        """
        CGSystem.mask_pdb(inpdb=self.syspdb, outpdb=self.trjpdb, mask=mask)
        CGSystem.make_index_by_chain(self.wdir, inpdb=self.trjpdb,  chains=self.chains, outndx=self.trjndx)
        
    @staticmethod
    @cli.from_wdir
    def make_index_by_chain(wdir, inpdb='mdc.pdb', outndx='mdc.ndx', chains=[]):
        """
        Makes a GROMACS index file containing required chains
        """
        chains = sort_uld(chains)
        commands = [f'chain {x}\n' for x in chains]
        clinput = 'case \nkeep 0\n' + ' '.join(commands) + 'q\n'
        cli.run('gmx_mpi make_ndx', f=inpdb, o=outndx, clinput=clinput)
    
    def pull_runs_files(self, fdir, fname):
        """
        Gets files from each run by name if the files exist
        
        Args:
            fname(str): Name of the file to read
               
        Output: 
            files(list): list of the fpaths
        """
        runs = [self.initmd(run) for run in self.mdruns]
        rundirs = [run.rundir for run in runs]
        files = [os.path.join(rundir, fdir, fname) for rundir in rundirs]
        files = [f for f in files if os.path.exists(f)]
        return files
        
    def get_mean_sem(self, files, outfname, col=1):
        """
        Calculates the mean and the standard error of mean for a metric
        of all runs and save them to self.datdir
        
        Args:
            fname(str): Name of the metric file
               
        Output: 
            data
        """
        dfs = []
        for file in files:
            df = pd.read_csv(file, sep='\\s+', header=None)
            dfs.append(df)
        datas = [df[col] for df in dfs]
        x = df[0]
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        df = pd.DataFrame({'x':x, 'mean':mean, 'sem':sem})
        fpath = os.path.join(self.datdir, outfname)
        df.to_csv(fpath, index=False, header=False, float_format='%.3E', sep=',')
        return x, mean, sem
        
    def get_mean_sem_2d(self, files, out_fname, out_errname):
        """
        Calculates the mean and the standard error of mean for a metric
        of all runs and save them to self.datdir
        
        Args:
            fname(str): Name of the metric file
               
        Output: 
            data
        """
        dfs = []
        for file in files:
            df = pd.read_csv(file, sep='\\s+', header=None)
            dfs.append(df)
        datas = dfs
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        df = pd.DataFrame(mean)
        fpath = os.path.join(self.datdir, out_fname)
        df.to_csv(fpath, index=False, header=False, float_format='%.3E', sep=',')
        df = pd.DataFrame(sem)
        fpath = os.path.join(self.datdir, out_errname)
        df.to_csv(fpath, index=False, header=False, float_format='%.3E', sep=',')
        return mean, sem
        

################################################################################
# MDRun class
################################################################################   

class MDRun(CGSystem):
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
        
    def empp(self, **kwargs):
        """
        Runs 'gmx grompp' preprocessing for energy minimization
        """
        kwargs.setdefault('f', os.path.join(self.mdpdir, 'em.mdp'))
        kwargs.setdefault('c', self.sysgro)
        kwargs.setdefault('r', self.sysgro)
        kwargs.setdefault('p', self.systop)
        kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', 'em.tpr')
        cli.gmx_grompp(self.rundir, **kwargs)
        
    def hupp(self, **kwargs):
        """
        Runs 'gmx grompp' preprocessing for heat up
        """
        kwargs.setdefault('f', os.path.join(self.mdpdir, 'hu.mdp'))
        kwargs.setdefault('c', 'em.gro')
        kwargs.setdefault('r', 'em.gro')
        kwargs.setdefault('p', self.systop)
        kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', 'hu.tpr')
        cli.gmx_grompp(self.rundir, **kwargs)

    def eqpp(self, **kwargs):
        """
        Runs 'gmx grompp' preprocessing for equilibration
        """
        kwargs.setdefault('f', os.path.join(self.mdpdir, 'eq.mdp'))
        kwargs.setdefault('c', 'hu.gro')
        kwargs.setdefault('r', 'hu.gro')
        kwargs.setdefault('p', self.systop)
        kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', 'eq.tpr')
        cli.gmx_grompp(self.rundir, **kwargs)
        
    def mdpp(self, grompp=True, **kwargs):
        """
        Runs 'gmx grompp' preprocessing for production
        """
        kwargs.setdefault('f', os.path.join(self.mdpdir, 'md.mdp'))
        kwargs.setdefault('c', 'eq.gro')
        kwargs.setdefault('r', 'eq.gro')
        kwargs.setdefault('p', self.systop)
        kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', 'md.tpr')
        cli.gmx_grompp(self.rundir, **kwargs)   
        
    def mdrun(self,**kwargs):
        """
        Runs 'gmx mdrun'
        """
        kwargs.setdefault('deffnm', 'md')
        kwargs.setdefault('nsteps', '-2')
        kwargs.setdefault('ntomp', '6')
        cli.gmx_mdrun(self.rundir, **kwargs) 
        
    def trjconv(self, clinput=None, **kwargs):
        """
        Runs 'gmx trjconv'
        """
        cli.gmx_trjconv(self.rundir, clinput=clinput, **kwargs)
         
    def rmsf(self, clinput=None, **kwargs):
        """
        Runs 'gmx rmsf' for RMSF calculation
        """
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        cli.gmx_rmsf(self.rundir, clinput=clinput, **kwargs)
         
    def rms(self, clinput=None, **kwargs):
        """
        Runs 'gmx rms' for RMSD calculation
        """        
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        cli.gmx_rms(self.rundir, clinput=clinput, **kwargs)
         
    def rdf(self, clinput=None, **kwargs):
        """
        Runs 'gmx rdf' for radial distribution function calculation
        """  
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        cli.gmx_rdf(self.rundir, clinput=clinput, **kwargs)  
        
    def cluster(self, clinput=None, **kwargs):
        """
        Runs 'gmx cluster' for clustering
        """  
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('s', '../traj.pdb')
        kwargs.setdefault('n', self.trjndx)
        cli.gmx_cluster(self.cludir, clinput=clinput, **kwargs) 
    
    def extract_cluster(self, clinput=None, **kwargs):
        """
        Runs 'gmx extract-cluster' to extract frames belonging to a cluster from the trajectory
        """  
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('clusters', 'cluster.ndx')
        cli.gmx_extract_cluster(self.cludir, clinput=clinput, **kwargs) 
        
    def covar(self, clinput=None, **kwargs):
        """
        Runs 'gmx covar' to calculate and diagonalize covariance matrix
        """  
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('s', '../traj.pdb')
        kwargs.setdefault('n', self.trjndx)
        cli.gmx_covar(self.covdir, clinput=clinput, **kwargs) 
        
    def anaeig(self, clinput=None, **kwargs):
        """
        Runs 'gmx anaeig' to analyze eigenvectors
        """  
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('s', '../traj.pdb')
        kwargs.setdefault('v', 'eigenvec.trr')
        cli.gmx_anaeig(self.covdir, clinput=clinput, **kwargs) 

    def make_edi(self, clinput=None, **kwargs):
        """
        Runs 'gmx make-edi' to prepare files for "essential dynamics"
        """         
        kwargs.setdefault('f', 'eigenvec.trr')        
        kwargs.setdefault('s', '../traj.pdb')
        cli.gmx_make_edi(self.covdir, clinput=clinput, **kwargs) 
        
    def get_covmats(self, **kwargs):
        """
        Calculates several covariance matrices by splitting the trajectory into 
        equal chunks defined by the given timestamps
        """
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('s', '../traj.pdb')
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        dd.calc_covmats(**kwargs)
        print('Finished calculating covariance matrices!', file=sys.stderr)
        os.chdir(bdir)

    def get_pertmats(self, **kwargs):
        """
        Calculates perturbation matrices from the covariance matrices
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        cov_files = [f for f in sorted(os.listdir()) if f.startswith('covmat')]
        for cov_file in cov_files:
            print(f'  Processing covariance matrix {cov_file}', file=sys.stderr)
            covmat = np.load(cov_file)
            print('  Calculating pertubation matrix', file=sys.stderr)
            pertmat = dd.calc_perturbation_matrix(covmat)
            pert_file = cov_file.replace('covmat', 'pertmat')
            print(f'  Saving pertubation matrix at {pert_file}', file=sys.stderr)
            np.save(pert_file, pertmat)
        print('Finished calculating perturbation matrices!', file=sys.stderr)
        os.chdir(bdir)
        
    def get_dfi(self, **kwargs):
        """
        Calculates DFI from perturbation matrices
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        pert_files = [f for f in sorted(os.listdir()) if f.startswith('pertmat')]
        for pert_file in pert_files:
            print(f'  Processing perturbation matrix {pert_file}', file=sys.stderr)
            pertmat = np.load(pert_file)
            print('  Calculating DFI', file=sys.stderr)
            dfi = dd.calc_dfi(pertmat)
            dfi_file = pert_file.replace('pertmat', 'dfi').replace('.npy', '.xvg')
            dfi_file = os.path.join('..', 'dci_dfi', dfi_file)
            print(f'  Saving DFI at {dfi_file}',  file=sys.stderr)
            dd.save_1d_data(dfi, fpath=dfi_file)
        print('Finished calculating DFIs!', file=sys.stderr)
        os.chdir(bdir)  
        
    def get_full_dci(self, asym=False):
        """
        Calculates full DCI matrix from perturbation matrices
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        pert_files = [f for f in sorted(os.listdir()) if f.startswith('pertmat')]
        for pert_file in pert_files:
            print(f'  Processing perturbation matrix {pert_file}', file=sys.stderr)
            pertmat = np.load(pert_file)
            print('  Calculating DCI', file=sys.stderr)
            dci_file = pert_file.replace('pertmat', f'dci').replace('.npy', '.xvg')
            ch_dci_file = os.path.join('..', 'dci_dfi', dci_file)
            dci = dd.calc_full_dci(pertmat, asym=asym)
            dd.save_2d_data(dci, fpath=dci_file)
        print('Finished calculating DCIs!', file=sys.stderr)
        os.chdir(bdir)

    def get_group_dci(self, groups=[], group_ids=[], asym=False):
        """
        Calculates DCI between given groups from perturbation matrices
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        pert_files = [f for f in sorted(os.listdir()) if f.startswith('pertmat')]
        for pert_file in pert_files:
            print(f'  Processing perturbation matrix {pert_file}', file=sys.stderr)
            pertmat = np.load(pert_file)
            print('  Calculating DCI', file=sys.stderr)
            dcis = dd.calc_group_molecule_dci(pertmat, groups=groups, asym=asym)
            for dci, group, group_id in zip(dcis, groups, group_ids):
                dci_file = pert_file.replace('pertmat', f'dci_{group_id}').replace('.npy', '.xvg')
                dci_file = os.path.join('..', 'dci_dfi', dci_file)
                print(f'  Saving DCI at {dci_file}',  file=sys.stderr)
                dd.save_1d_data(dci, fpath=dci_file)
            ch_dci_file = pert_file.replace('pertmat', f'ch_dci').replace('.npy', '.xvg')
            ch_dci_file = os.path.join('..', 'dci_dfi', ch_dci_file)
            ch_dci = dd.calc_group_group_dci(pertmat, groups=groups, asym=asym)
            dd.save_2d_data(ch_dci, fpath=ch_dci_file)
        print('Finished calculating DCIs!', file=sys.stderr)
        os.chdir(bdir) 
        
    def get_rmsf_by_chain(self, **kwargs):
        """
        Get RMSF by chain.
        """
        kwargs.setdefault('f', 'traj.xtc')
        kwargs.setdefault('s', 'traj.pdb')
        kwargs.setdefault('n', self.trjndx)
        kwargs.setdefault('res', 'no')
        kwargs.setdefault('fit', 'yes')
        for idx, chain in enumerate(self.chains):
            idx = idx + 1
            cli.gmx_rmsf(self.rundir, clinput=f'{idx}\n{idx}\n', 
                o=os.path.join(self.rmsdir, f'rmsf_{chain}.xvg'), **kwargs)
                
    def get_rmsd_by_chain(self, **kwargs):
        """
        Get RMSD by chain.
        """
        kwargs.setdefault('f', 'traj.xtc')
        kwargs.setdefault('s', 'traj.pdb')
        kwargs.setdefault('n', self.trjndx)
        for idx, chain in enumerate(self.chains):
            idx = idx + 1
            cli.gmx_rmsf(self.rundir, clinput=f'{idx}\n', 
                o=os.path.join(self.rmsdir, f'rmsf_{chain}.xvg'), **kwargs)
                
            
