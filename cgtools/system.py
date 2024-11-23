import os
import sys
import numpy as np
import pandas as pd
import shutil
import subprocess as sp
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO, Atom
import cli
from cli import from_wdir


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
        A list of chain ids. Either provide or look up in self.cgdir
        Returns:
            list: List of chain identifiers.
        """
        if self._chains:
            return self._chains
        if not os.path.isdir(self.cgdir):
            return self._chains
        for filename in sorted(os.listdir(self.cgdir)):
            if filename.endswith('.pdb'):
                chain_id = filename.split('.')[0].split('_')[1]
                self._chains.append(chain_id)
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
                # command = f'ln -sf {fpath} {out}' # > /dev/null 2>&
                # sp.run(command.split())
                shutil.copy(fpath, outpath)
        
    def clean_inpdb(self, **kwargs):
        print("Cleaning the PDB", file=sys.stderr)
        from pdbtools import prepare_aa_pdb
        in_pdb = self.inpdb
        out_pdb = in_pdb.replace('.pdb', '_clean.pdb')
        prepare_aa_pdb(in_pdb, out_pdb, **kwargs)  
        
    def clean_proteins(self, **kwargs):
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
        
    def martinize_proteins(self, **kwargs):
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
                    f.write(f'#include "{self.topdir}/{filename}"\n')
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
        outpdb = kwargs.pop('outpdb', self.mdcpdb)
        outndx = kwargs.pop('outndx', self.mdcndx)
        CGSystem.mask_pdb(inpdb=self.syspdb, outpdb=outpdb, mask=mask)
        CGSystem.make_index_by_chain(self.wdir, chains=self.chains, outndx=outndx)
        
        
    def make_trj_pdb_ndx(self, mask=['BB', 'BB2', ], **kwargs):
        """
        Makes a pdb file of masked selected atoms and a ndx file with separated chains for this pdb
        Default is to select backbone atoms for proteins and RNA
        """
        outpdb = kwargs.pop('outpdb', self.trjpdb)
        outndx = kwargs.pop('outndx', self.trjndx)
        CGSystem.mask_pdb(inpdb=self.syspdb, outpdb=outpdb, mask=mask)
        CGSystem.make_index_by_chain(self.wdir, inpdb=self.trjpdb, outndx=outndx, chains=self.chains,)
        
    @staticmethod
    @from_wdir
    def make_index_by_chain(wdir, inpdb='mdc.pdb', outndx='mdc.ndx', chains=[]):
        """
        Makes a GROMACS index file containing required chains
        """
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
        
    def get_mean_sem(self, fdir, fname, col=1):
        """
        Calculates the mean and the standard error of mean for a metric
        of all runs and save them to self.datdir
        
        Args:
            fname(str): Name of the metric file
               
        Output: 
            data
        """
        files = self.pull_runs_files(fdir, fname)
        dfs = []
        for file in files:
            df = pd.read_csv(file, sep='\\s+', header=None)
            dfs.append(df)
        datas = [df[col] for df in dfs]
        x = df[0]
        mean = np.average(datas, axis=0)
        sem = np.std(datas, axis=0) / np.sqrt(len(datas))
        df = pd.DataFrame({'x':x, 'mean':mean, 'sem':sem})
        fpath = os.path.join(self.datdir, fname.replace('xvg', 'csv'))
        df.to_csv(fpath, index=False, header=False, float_format='%.3E', sep=',')
        return x, mean, sem
        

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
        self.rundir = os.path.join(self.mddir, self.runname)
        self.rmsdir = os.path.join(self.rundir, 'rms_analysis')
        self.covdir = os.path.join(self.rundir, 'cov_analysis')
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
        os.makedirs(self.pngdir, exist_ok=True)
        
    def empp(self, **kwargs):
        """
        Runs 'gmx grompp' preprocessing for energy minimization
        """
        kwargs.setdefault('f', os.path.join(self.mdpdir, 'em.mdp'))
        kwargs.setdefault('c', self.sysgro)
        kwargs.setdefault('r', self.sysgro)
        kwargs.setdefault('p', self.systop)
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
        
    def covmat(self, **kwargs):
        # kwargs.setdefault('f', 'mdc.xtc')
        # kwargs.setdefault('s', 'mdc.pdb')
        from dci_dfi import calculate_covariance_matrix
        f = os.path.join(self.rundir, 'mdc.xtc')
        s = self.mdcpdb
        o = os.path.join(self.covdir, 'cov.npy')
        calculate_covariance_matrix(f, s, o)
        
    def pertmat(self, **kwargs):
        from dci_dfi import parse_covar_dat, get_perturbation_matrix
        covdat = os.path.join(self.covdir, 'covar.dat')
        print('Reading covariance matrix', file=sys.stderr)
        cov, resnum = parse_covar_dat(covdat)
        print('Calculating pertubation matrix', file=sys.stderr)
        pertmat = get_perturbation_matrix(cov, resnum)
        print('Saving pertubation matrix', file=sys.stderr)
        np.save(os.path.join(self.covdir, 'pertmat.npy'), pertmat)
        
    def get_rmsf_by_chain(self, **kwargs):
        """
        Get RMSF by chain.
        """
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        for idx, chain in enumerate(self.chains):
            idx = idx + 1
            cli.gmx_rmsf(self.rundir, clinput=f'{idx}\n', 
                o=os.path.join(self.rmsdir, f'rmsf_{chain}.xvg'), **kwargs)
                
    def get_rmsd_by_chain(self, **kwargs):
        """
        Get RMSD by chain.
        """
        kwargs.setdefault('f', 'mdc.xtc')
        kwargs.setdefault('s', 'mdc.pdb')
        kwargs.setdefault('n', self.mdcndx)
        for chain in self.chains:
            cli.gmx_rms(self.rundir, clinput=f'ch{chain}\nch{chain}\n', 
                o=os.path.join(self.rmsdir, f'rmsd_{chain}.xvg'), **kwargs)
                
            
