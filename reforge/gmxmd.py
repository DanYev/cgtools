import os
import sys
import logging
import importlib.resources
import MDAnalysis as mda
import numpy as np
import pandas as pd
import shutil
import subprocess as sp
import reforge
from reforge import cli, mdm, pdbtools, io
from reforge.pdbtools import AtomList
from reforge.utils import cd, clean_dir, logger
     

################################################################################
# GMX system class
################################################################################   

class gmxSystem:
    """
    Class to set up and analyze protein-nucliotide-lipid systems for MD with GROMACS
    Almost all the attributes are paths to files and directories needed to set up and run the MD
    """    
    MDATDIR = importlib.resources.files("reforge") / "martini" / "data" 
    MMDPDIR= importlib.resources.files("reforge") / "martini" / "data" / "mdp"
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
        self.solupdb    = os.path.join(self.wdir, 'solute.pdb')
        self.syspdb     = os.path.join(self.wdir, 'system.pdb')
        self.sysgro     = os.path.join(self.wdir, 'system.gro')        
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

    @property
    def chains(self):
        atoms = io.pdb2atomlist(self.inpdb)
        chains = sort_upper_lower_digit(set(atoms.chids))
        return chains

    def prepare_files(self):
        """
        Creates the necessary directories for the simulation and copies the 
        relevant input files to the working directory.
        """
        logger.info("Preparing files and directories")
        os.makedirs(self.prodir, exist_ok=True)
        os.makedirs(self.nucdir, exist_ok=True)
        os.makedirs(self.topdir, exist_ok=True)
        os.makedirs(self.mapdir, exist_ok=True)
        os.makedirs(self.mdpdir, exist_ok=True)
        os.makedirs(self.cgdir,  exist_ok=True)
        os.makedirs(self.grodir, exist_ok=True)
        os.makedirs(self.datdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        for file in os.listdir(self.MMDPDIR):
            if file.endswith('.mdp'):
                fpath = os.path.join(self.MMDPDIR, file)
                outpath = os.path.join(self.mdpdir, file)
                shutil.copy(fpath, outpath)
        shutil.copy(os.path.join(self.MDATDIR, 'water.gro'), self.wdir)
        shutil.copy(os.path.join(self.MDATDIR, 'atommass.dat'), self.wdir)
        for file in os.listdir(self.MITPDIR):
            if file.endswith('.itp'):
                fpath = os.path.join(self.MITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                shutil.copy(fpath, outpath)
                
    def sort_input_pdb(self, in_pdb="inpdb.pdb",):
        """
        Sort and rename atoms and chains
        """
        with cd(self.wdir):
            pdbtools.sort_pdb(in_pdb, self.inpdb)
                       
    def clean_pdb_mm(self, in_pdb=None, **kwargs):
        """
        Cleans starting PDB file using PDBfixer by OpenMM
        """
        logger.info("Cleaning the PDB...")
        if not in_pdb:
            in_pdb = self.inpdb
        pdbtools.clean_pdb(in_pdb, in_pdb, **kwargs)  

    def clean_pdb_gmx(self, in_pdb=None, **kwargs):
        """
        Cleans PDB files using pdb2gmx from GROMACS
        gmx pdb2gmx [-f [<.gro/.g96/...>]] [-o [<.gro/.g96/...>]] [-p [<.top>]]
        Needed to get Go-Maps from the web-server 
        http://info.ifpan.edu.pl/~rcsu/rcsu/index.html
        """
        logger.info("Cleaning the PDB...")
        if not in_pdb:
            in_pdb = self.inpdb
        with cd(self.wdir):    
            cli.gmx('pdb2gmx', f=in_pdb, o=in_pdb, **kwargs)
            
    def split_chains(self):
        """
        Cleans a separate PDB file for each chain in the initial structure
        """
        def it_is_nucleotide(atoms): # check if it's RNA or DNA based on residue name
            return atoms.resnames[0] in self.NUC_RESNAMES
        logger.info("Splitting chains...")
        system = pdbtools.pdb2system(self.inpdb)
        for chain in system.chains():
            atoms = chain.atoms
            if it_is_nucleotide(atoms):
                out_pdb = os.path.join(self.nucdir, f'chain_{chain.chid}.pdb')
            else:
                out_pdb = os.path.join(self.prodir, f'chain_{chain.chid}.pdb')
            atoms.write_pdb(out_pdb)
                
    def clean_chains_mm(self, **kwargs):
        """
        Cleans PDB files using PDBfixer by OpenMM
        """
        kwargs.setdefault('add_missing_atoms', True)
        kwargs.setdefault('add_hydrogens', True)
        kwargs.setdefault('pH', 7.0)
        logger.info("Cleaning chain PDBs")
        files = [os.path.join(self.prodir, f) for f in os.listdir(self.prodir)]
        files += [os.path.join(self.nucdir, f) for f in os.listdir(self.nucdir)]
        files = sorted(files)
        for file in files:
            pdbtools.clean_pdb(file, file, **kwargs)
            new_chain_id = file.split('chain_')[1][0]
            pdbtools.rename_chain_in_pdb(file, new_chain_id) 

    def clean_chains_gmx(self, **kwargs):
        """
        Cleans PDB files using pdb2gmx from GROMACSv
        gmx pdb2gmx [-f [<.gro/.g96/...>]] [-o [<.gro/.g96/...>]] [-p [<.top>]]
        Needed to get Go-Maps from the web-server 
        http://info.ifpan.edu.pl/~rcsu/rcsu/index.html
        """
        logger.info("Cleaning chain PDBs")
        files = [os.path.join(self.prodir, f) for f in os.listdir(self.prodir) if not f.startswith('#')]
        files += [os.path.join(self.nucdir, f) for f in os.listdir(self.nucdir) if not f.startswith('#')]
        files = sorted(files)
        with cd(self.wdir): 
            for file in files:
                new_chain_id = file.split('chain_')[1][0]
                cli.gmx('pdb2gmx', f=file, o=file, **kwargs)
                pdbtools.rename_chain_and_histidines_in_pdb(file, new_chain_id)  
            clean_dir(self.prodir)
            clean_dir(self.nucdir)
        
    def get_go_maps(self, append=False):
        """
        Get go contact maps for proteins using RCSU server
        """
        print('Getting GO-maps', file=sys.stderr)
        from reforge.martini import getgo
        pdbs = sorted([os.path.join(self.prodir, file) for file in os.listdir(self.prodir)])
        map_names = [f.replace('pdb', 'map') for f in os.listdir(self.prodir)]
        if append: # Filter out existing maps
            pdbs = [pdb for pdb, amap in zip(pdbs, map_names) if amap not in os.listdir(self.mapdir)]
        if pdbs:
            getgo.get_go(self.mapdir, pdbs)
        else:
            print('Maps already there', file=sys.stderr)
        
    def martinize_proteins_go(self, append=False, **kwargs):
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
        logger.info("Working on proteins")
        from reforge.martini.martini_tools import martinize_go
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace('pdb', 'itp') for f in pdbs]
        if append: # Filter out existing topologies
            pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        else:
            clean_dir(self.topdir, 'go_*.itp')
        # Make itp files to dump all the virtual CA's parameters into
        file = os.path.join(self.topdir, 'go_atomtypes.itp')
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                f.write(f'[ atomtypes ]\n')
        file = os.path.join(self.topdir, 'go_nbparams.itp')
        if not os.path.isfile(file):
            with open(file, 'w') as f:        
                f.write(f'[ nonbond_params ]\n')       
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            name = file.split('.')[0]
            go_map = os.path.join(self.mapdir, f'{name}.map')
            martinize_go(self.wdir, self.topdir, in_pdb, cg_pdb, name=name, **kwargs)
        clean_dir(self.cgdir)
        clean_dir(self.wdir)
        clean_dir(self.wdir, '*.itp')
         
    def martinize_proteins_en(self, append=False, **kwargs):
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
        logger.info("Working on proteins")
        from .martini.martini_tools import martinize_en
        pdbs = sorted(os.listdir(self.prodir))
        itps = [f.replace('pdb', 'itp') for f in pdbs]
        if append: # Filter out existing topologies
            pdbs = [pdb for pdb, itp in zip(pdbs, itps) if itp not in os.listdir(self.topdir)]
        for file in pdbs:
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            new_itp = os.path.join(self.wdir, 'molecule_0.itp')
            updated_itp = os.path.join(self.topdir, file.replace('pdb', 'itp'))
            new_top = os.path.join(self.wdir, 'protein.top')
            martinize_en(self.wdir, in_pdb, cg_pdb,  **kwargs) 
            # Replace 'molecule_0' in the itp with the file name
            with open(new_itp, "r", encoding="utf-8") as f:
                content = f.read()
            updated_content = content.replace('molecule_0', f'{file[:-4]}', 1)
            with open(updated_itp, "w", encoding="utf-8") as f:
                f.write(updated_content)
            os.remove(new_top)
        clean_dir(self.cgdir)
        clean_dir(self.wdir)
    
    def martinize_nucleotides(self, append=False, **kwargs):
        logger.info("Working on nucleotides")
        from .martini.martini_tools import martinize_nucleotide
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
        clean_dir(self.cgdir)
        clean_dir(self.wdir)    

    def martinize_rna(self, append=False, **kwargs):
        logger.info("Working on nucleotides")
        from reforge.martini.martini_tools import martinize_rna
        for file in os.listdir(self.nucdir):
            molname = file.split('.')[0]
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            cg_itp = os.path.join(self.topdir, molname + '.itp')
            try:
                martinize_rna(self.wdir, f=in_pdb, os=cg_pdb, ot=cg_itp, mol=molname, **kwargs)
            except Exception as e:
                sys.exit(f"Could not coarse-grain {in_pdb}: {e}")
            

    def make_cgpdb_file(self, add_resolved_ions=False, **kwargs):
        logger.info("Merging CG PDBs")
        with cd(self.wdir):
            cg_pdb_files = os.listdir(self.cgdir)
            cg_pdb_files = sort_upper_lower_digit(cg_pdb_files)
            cg_pdb_files = [os.path.join(self.cgdir, fname) for fname in cg_pdb_files]
            all_atoms = AtomList()
            for file in cg_pdb_files:
                atoms = pdbtools.pdb2atomlist(file)
                all_atoms.extend(atoms)
            if add_resolved_ions:
                ions = pdbtools.pdb2atomlist(self.ionpdb)
                all_atoms.extend(ions)
            all_atoms.renumber()    
            all_atoms.write_pdb(self.solupdb)
            cli.gmx('editconf', f=self.solupdb, o=self.solupdb, **kwargs)   
        
    def make_martini_topology_file(self, add_resolved_ions=False, prefix='chain'):
        logger.info("Writing CG Topology")
        itp_files = [f for f in os.listdir(self.topdir) if f.startswith(prefix) and f.endswith('.itp')] 
        itp_files = sort_upper_lower_digit(itp_files)
        with open(self.systop, 'w') as f:
            # Include section
            f.write(f'#define GO_VIRT"\n')
            f.write(f'#define RUBBER_BANDS\n')
            f.write(f'#include "topol/martini_v3.0.0.itp"\n')
            f.write(f'#include "topol/martini_v3.0.0_rna.itp"\n')
            f.write(f'#include "topol/martini_ions.itp"\n')
            if 'go_atomtypes.itp' in os.listdir(self.topdir):
                f.write(f'#include "topol/go_atomtypes.itp"\n')
                f.write(f'#include "topol/go_nbparams.itp"\n')
            f.write(f'#include "topol/martini_v3.0.0_solvents_v1.itp"\n') 
            f.write(f'#include "topol/martini_v3.0.0_phospholipids_v1.itp"\n')
            f.write(f'#include "topol/martini_v3.0.0_ions_v1.itp"\n')
            f.write(f'\n')
            for filename in itp_files:
                f.write(f'#include "topol/{filename}"\n')
            # System name
            f.write(f'\n[ system ]\n')
            f.write(f'Martini system for {self.sysname}\n') 
            # Molecules
            f.write('\n[molecules]\n')
            f.write('; name\t\tnumber\n')
            for filename in itp_files:
                molecule_name = os.path.splitext(filename)[0]
                f.write(f'{molecule_name}\t\t1\n')
            # Ions
            if add_resolved_ions:
                ions = self.count_resolved_ions() 
                for ion, count in ions.items():
                    if count > 0:
                        f.write(f'{ion}    {count}\n')

    def make_gro_file(self, d=1.25, bt='dodecahedron'):
        with cd(self.wdir):
            cg_pdb_files = os.listdir(self.cgdir)
            cg_pdb_files = sort_upper_lower_digit(cg_pdb_files)
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
        with cd(self.wdir):
            cli.gmx_solvate(**kwargs)
        
    def find_resolved_ions(self, mask=['MG', 'ZN', 'K']):
        mask_atoms(self.inpdb, 'ions.pdb', mask=mask)
        
    def count_resolved_ions(self, ions=['MG', 'ZN', 'K']): # does NOT work for CA atoms for now
        counts = {ion: 0 for ion in ions}
        with open(self.syspdb, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    current_ion = line[12:16].strip()
                    if current_ion in ions:
                        counts[current_ion] += 1
        return counts
        
    def add_bulk_ions(self, conc=0.15, pname='NA', nname='CL'): 
        with cd(self.wdir):
            command = f'gmx_mpi grompp -f mdp/ions.mdp -c {self.syspdb} -p system.top -o ions.tpr'
            sp.run(command.split())
            command = f'gmx_mpi genion -s ions.tpr -p system.top -conc {conc} -neutral -pname {pname} -nname {nname} -o {self.syspdb}'
            sp.run(command.split(), input='W\n', text=True)
            cli.gmx_editconf(f=self.syspdb, o=self.sysgro)
        clean_dir(self.wdir, pattern='ions.tpr')       

    def make_sys_ndx(self, backbone_atoms=["CA", "P", "C1'"]): # ["BB", "BB1", "BB3"]
        logger.info("Making Index File")
        system = pdbtools.pdb2atomlist(self.syspdb)
        solute = pdbtools.pdb2atomlist(self.solupdb)
        solvent = AtomList(system[len(solute):])
        backbone = solute.mask(backbone_atoms, mode='name')
        system.write_ndx(self.sysndx, header=f'[ System ]', append=False, wrap=15) 
        solute.write_ndx(self.sysndx, header=f'[ Solute ]', append=True, wrap=15) 
        backbone.write_ndx(self.sysndx, header=f'[ Backbone ]', append=True, wrap=15) 
        solvent.write_ndx(self.sysndx, header=f'[ Solvent ]', append=True, wrap=15) 
        # Chains
        chids = sorted(set(solute.chids))
        for chid in chids:
            chain = solute.mask(chid, mode='chid')
            chain.write_ndx(self.sysndx, header=f'[ chain_{chid} ]', append=True, wrap=15) 
        logger.info(f"Written index to {self.sysndx}")

    def get_mean_sem(self, pattern='dfi*.npy'):
        logger.info(f"Calculating averages and errors from {pattern}")
        files = io.pull_files(self.mddir, pattern)
        datas = [np.load(file) for file in files]
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        file_mean = os.path.join(self.datdir, pattern.split('*')[0] + '_av.npy')
        file_err = os.path.join(self.datdir, pattern.split('*')[0] + '_err.npy')
        np.save(file_mean, mean)
        np.save(file_err, sem)
        # pd.DataFrame(mean).to_csv(file_mean, header=None, index=False) 
        # pd.DataFrame(sem).to_csv(file_err, header=None, index=False) 

    def get_td_averages(self, fname, loop=True):
        """
        Need to loop for big arrays
        """
        system = gmxSystem(sysdir, sysname)  
        logger.info('Getting time-dependent averages')
        files = io.pull_files(system.mddir, fname)
        if loop:
            logger.info(f'Processing {files[0]}')
            average = np.load(files[0])
            for f in files[1:]:
                logger.info(f'Processing {f}')
                arr = np.load(f)
                average += arr
            average /= len(files)
        else:
            arrays = [np.load(f) for f in files]
            average = np.average(arrays, axis=0)
        np.save(os.path.join(system.datdir, fname), average) 
        logger.info('Done!')
        return average     

    def get_averages(self, rmsf=False, dfi=True, dci=True, ): 
        all_files = io.pull_all_files(self.mddir)
        if rmsf:  # RMSF
            files = io.filter_files(all_files, sw='rmsf.', ew='.xvg')
            system.get_mean_sem(files, f'rmsf.csv', col=1)
            # Chain RMSF
            for chain in system.chains:
                sw = f'rmsf_{chain}'
                files = io.filter_files(all_files, sw=sw, ew='.xvg')
                system.get_mean_sem(files, f'{sw}.csv', col=1)
        if dfi:  # DFI
            print(f'Processing DFI', file=sys.stderr )
            files = io.filter_files(all_files, sw='dfi', ew='.xvg')
            system.get_mean_sem(files, f'dfi.csv', col=1)
        if dci: # DCI
            print(f'Processing DCI', file=sys.stderr )
            files = io.filter_files(all_files, sw='dci', ew='.xvg')
            system.get_mean_sem_2d(files, out_fname=f'dci.csv', out_errname=f'dci_err.csv')
            files = io.filter_files(all_files, sw='asym', ew='.xvg')
            system.get_mean_sem_2d(files, out_fname=f'dci.csv', out_errname=f'dci_err.csv')   

    def initmd(self, runname):
        mdrun = MDRun(self.sysdir, self.sysname, runname)
        # self._mdruns.append(mdrun.runname)
        return mdrun                    

################################################################################
# MDRun class
################################################################################   

class MDRun(gmxSystem):
    """
    Run molecular dynamics (MD) simulation using the specified input files.
    This method runs the MD simulation by calling an external simulation software, 
    such as GROMACS, with the provided input files.
    """
    
    def __init__(self, sysdir, sysname, runname):
        """
        Initializes required directories and files.
        
        Args:
            sysdir (str): Directory for the system files.
            sysname (str): Name of the system.       
            runname (str): Name of the MD run.

            kwargs: Additional keyword arguments.
        """
        super().__init__(sysdir, sysname)
        self.runname = runname
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, 'rms_analysis')
        self.covdir = os.path.join(self.rundir, 'cov_analysis')
        self.lrtdir = os.path.join(self.rundir, 'lrt_analysis')
        self.cludir = os.path.join(self.rundir, 'clusters')
        self.pngdir = os.path.join(self.rundir, 'png')
        self.str = os.path.join(self.rundir, 'mdc.pdb') # Structure
        self.trj = os.path.join(self.rundir, 'mdc.trr') # Trajectory
              
    def prepare_files(self):
        """
        Create necessary directories.
        """
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(self.rmsdir, exist_ok=True)
        os.makedirs(self.cludir, exist_ok=True)
        os.makedirs(self.covdir, exist_ok=True)
        os.makedirs(self.lrtdir, exist_ok=True)
        os.makedirs(self.pngdir, exist_ok=True)
        shutil.copy('atommass.dat', os.path.join(self.rundir, 'atommass.dat'))
        
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
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)
        
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
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)

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
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)
        
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
        with cd(self.rundir):
            cli.gmx_grompp(**kwargs)   
        
    def mdrun(self,**kwargs):
        """
        Runs 'gmx mdrun'
        """
        kwargs.setdefault('deffnm', 'md')
        kwargs.setdefault('nsteps', '-2')
        kwargs.setdefault('ntomp', '8')
        with cd(self.rundir):
            cli.gmx_mdrun(**kwargs) 
        
    def trjconv(self, clinput=None, **kwargs):
        """
        Runs 'gmx trjconv'
        """
        with cd(self.rundir):
            cli.gmx_trjconv(clinput=clinput, **kwargs)
         
    def rmsf(self, clinput=None, **kwargs):
        """
        Runs 'gmx rmsf' for RMSF calculation
        """
        xvg_file = os.path.join(self.rmsdir, 'rmsf.xvg')
        npy_file = os.path.join(self.rmsdir, 'rmsf.npy')
        kwargs.setdefault('s', self.str)      
        kwargs.setdefault('f', self.trj)
        # kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', xvg_file)
        with cd(self.rmsdir):
            cli.gmx_rmsf(clinput=clinput, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[1])
         
    def rmsd(self, clinput=None, **kwargs):
        """
        Runs 'gmx rms' for RMSD calculation
        """ 
        xvg_file = os.path.join(self.rmsdir, 'rmsd.xvg')
        npy_file = os.path.join(self.rmsdir, 'rmsd.npy') 
        kwargs.setdefault('s', self.str)      
        kwargs.setdefault('f', self.trj)
        # kwargs.setdefault('n', self.sysndx)
        kwargs.setdefault('o', xvg_file)
        with cd(self.rmsdir):
            cli.gmx_rms(clinput=clinput, **kwargs)
            io.xvg2npy(xvg_file, npy_file, usecols=[0, 1])
         
    def rdf(self, clinput=None, **kwargs):
        """
        Runs 'gmx rdf' for radial distribution function calculation
        """  
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        with cd(self.rmsdir):
            cli.gmx_rdf(clinput=clinput, **kwargs)  
        
    def cluster(self, clinput=None, **kwargs):
        """
        Runs 'gmx cluster' for clustering
        """  
        kwargs.setdefault('s', self.str)      
        kwargs.setdefault('f', self.trj)
        # kwargs.setdefault('n', self.sysndx)
        with cd(self.cludir):
            cli.gmx_cluster(clinput=clinput, **kwargs) 
    
    def extract_cluster(self, clinput=None, **kwargs):
        """
        Runs 'gmx extract-cluster' to extract frames belonging to a cluster from the trajectory
        """  
        # kwargs.setdefault('s', self.str)      
        kwargs.setdefault('f', self.trj)
        kwargs.setdefault('clusters', 'cluster.ndx')
        with cd(self.cludir):
            cli.gmx_extract_cluster(clinput=clinput, **kwargs) 
        
    def covar(self, clinput=None, **kwargs):
        """
        Runs 'gmx covar' to calculate and diagonalize covariance matrix
        """  
        kwargs.setdefault('f', '../traj.xtc')
        kwargs.setdefault('s', '../traj.pdb')
        kwargs.setdefault('n', self.trjndx)
        with cd(self.covdir):
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
        
    def get_covmats(self, u=None, ag=None, sample_rate=1, b=50000, e=1000000, n=10, outtag='covmat'):
        """
        Calculates several covariance matrices by splitting the trajectory into 
        equal chunks defined by the given timestamps
        """
        logger.info('Calculating covariance matrices...')
        if not u:
            u = mda.Universe(self.str, self.trj, in_memory=True)
        if not ag: # Select the backbone atoms
            ag = u.atoms.select_atoms("name BB or name BB1 or name BB3") 
            if not ag:
                ag = u.atoms.select_atoms("name CA or name P or name C1'")
        positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e) 
        mdm.calc_and_save_covmats(positions, outdir=self.covdir, n=n, outtag=outtag) 
        logger.info('Finished calculating covariance matrices!')

    def get_pertmats(self, intag='covmat', outtag='pertmat', **kwargs):
        """
        Calculates perturbation matrices from the covariance matrices
        """
        with cd(self.covdir):
            cov_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for cov_file in cov_files:
                logger.info(f'  Processing covariance matrix {cov_file}')
                covmat = np.load(cov_file)
                logger.info('  Calculating pertubation matrix')
                pertmat = mdm.perturbation_matrix(covmat)
                pert_file = cov_file.replace(intag, outtag)
                logger.info(f'  Saving pertubation matrix at {pert_file}')
                np.save(pert_file, pertmat)
        logger.info('Finished calculating perturbation matrices!')
        
    def get_dfi(self, intag='pertmat', outtag='dfi', **kwargs):
        """
        Calculates DFI from perturbation matrices
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info(f'  Processing perturbation matrix {pert_file}')
                pertmat = np.load(pert_file)
                logger.info('  Calculating DFI')
                dfi = mdm.dfi(pertmat)
                dfi_file = pert_file.replace(intag, outtag)
                dfi_file = os.path.join(self.covdir, dfi_file)
                np.save(dfi_file, dfi)
                logger.info(f'  Saved DFI at {dfi_file}')
        logger.info('Finished calculating DFIs!')
        
    def get_dci(self, intag='pertmat', outtag='dci', asym=False):
        """
        Calculates full DCI matrix from perturbation matrices
        """
        with cd(self.covdir):
            pert_files = [f for f in sorted(os.listdir()) if f.startswith(intag)]
            for pert_file in pert_files:
                logger.info(f'  Processing perturbation matrix {pert_file}')
                pertmat = np.load(pert_file)
                logger.info('  Calculating DCI')
                dci_file = pert_file.replace(intag, outtag)
                dci_file = os.path.join(self.covdir, dci_file)
                dci = mdm.dci(pertmat, asym=asym)
                np.save(dci_file, dci)
                logger.info(f'  Saved DCI at {dci_file}')
        logger.info('Finished calculating DCIs!')

    def get_group_dci(self, groups=[], labels=[], asym=False):
        """
        Calculates DCI between given groups from perturbation matrices
        """
        bdir = os.getcwd()
        os.chdir(self.covdir)
        logger.info(f'Working dir: {self.covdir}')
        pert_files = [f for f in sorted(os.listdir()) if f.startswith('pertmat')]
        for pert_file in pert_files:
            logger.info(f'  Processing perturbation matrix {pert_file}')
            pertmat = np.load(pert_file)
            logger.info('  Calculating DCI')
            dcis = mdm.group_molecule_dci(pertmat, groups=groups, asym=asym)
            for dci, group, group_id in zip(dcis, groups, labels):
                dci_file = pert_file.replace('pertmat', f'gdci_{group_id}')
                dci_file = os.path.join(self.covdir, dci_file)
                np.save(dci_file, dci)
                logger.info(f'  Saved DCI at {dci_file}')
            ch_dci_file = pert_file.replace('pertmat', f'ggdci')
            ch_dci_file = os.path.join(self.covdir, ch_dci_file)
            ch_dci = mdm.group_group_dci(pertmat, groups=groups, asym=asym)
            np.save(ch_dci_file, ch_dci)
            logger.info(f'  Saved DCI at {ch_dci_file}')
        logger.info('Finished calculating DCIs!')
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
                

    def get_power_spectrum_xv(self, resp_ids=[], pert_ids=[], **kwargs):
        """
        Position-velocity correlation
        """
        kwargs.setdefault('f', '../traj.trr')
        kwargs.setdefault('s', '../traj.pdb')
        bdir = os.getcwd()
        os.chdir(self.covdir)
        print(f'Working dir: {self.covdir}', file=sys.stderr)
        mdm.calc_power_spectrum_xv(resp_ids, pert_ids, **kwargs)
        print('Finished calculating', file=sys.stderr)
        os.chdir(bdir) 
        
################################################################################
# Utils
################################################################################   

def sort_upper_lower_digit(alist):
    """
    Sorts characters in a list such that they appear in the following order: 
    uppercase letters first, then lowercase letters, followed by digits. 
    Helps with orgazing GROMACS multichain files
    """
    slist = sorted(alist, key=lambda x: (x.isdigit(), x.islower(), x.isupper(), x))
    return slist                      
