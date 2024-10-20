import os
import sys
import numpy as np
import shutil
import subprocess as sp
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO, Atom
from itertools import chain
import cli


class System:
    """
    Class to set up and analyze protein-nucliotide-lipid systems
    """    
    
    nuc_resnames = ['A', 'C', 'G', 'U', 'RA3', 'RA5', 'RC3', 'RC5', 'RG3', 'RG5', 'RU3', 'RU5']
    
    def __init__(self, sysdir, sysname, pdb, **kwargs):
        """
        Init
        """
                
        self.sysname    = sysname
        self.sysdir     = os.path.abspath(sysdir)
        self.wdir       = os.path.join(self.sysdir, sysname)
        self.syspdb     = os.path.join(self.wdir, pdb)
        self.sysgro     = os.path.join(self.wdir, 'system.gro')
        self.syscg      = os.path.join(self.wdir, 'system.pdb')
        self.systop     = os.path.join(self.wdir, 'system.top')
        self.prodir     = os.path.join(self.wdir, 'proteins')
        self.nucdir     = os.path.join(self.wdir, 'nucleotides')
        self.topdir     = os.path.join(self.wdir, 'topol')
        self.mapdir     = os.path.join(self.wdir, 'map')
        self.mdpdir     = os.path.join(self.wdir, 'mdp')
        self.cgdir      = os.path.join(self.wdir, 'cgpdb')
        self.grodir     = os.path.join(self.wdir, 'gro')
        self.chains     = self._get_chain_ids()
        self.mdruns     = []
        
    def prepare_files(self):
        print('Preparing files and directories')
        os.makedirs(self.prodir, exist_ok=True)
        os.makedirs(self.nucdir, exist_ok=True)
        os.makedirs(self.topdir, exist_ok=True)
        os.makedirs(self.mapdir, exist_ok=True)
        os.makedirs(self.mdpdir, exist_ok=True)
        os.makedirs(self.cgdir,  exist_ok=True)
        os.makedirs(self.grodir, exist_ok=True)
        DATADIR = 'cgtools/data'
        for file in os.listdir(DATADIR):
            fpath = os.path.join(DATADIR, file)
            outpath = os.path.join(self.mdpdir, file)
            shutil.copy(fpath, outpath)
        ITPDIR = 'cgtools/itp'
        for file in os.listdir(ITPDIR):
            if file.endswith('.itp'):
                fpath = os.path.join(ITPDIR, file)
                outpath = os.path.join(self.topdir, file)
                # command = f'ln -sf {fpath} {out}' # > /dev/null 2>&
                # sp.run(command.split())
                shutil.copy(fpath, outpath)
        
    def clean_pdb(self):
        from pdbtools import prepare_aa_pdb
        in_pdb = self.syspdb
        out_pdb = in_pdb.replace('.pdb', '_clean.pdb')
        prepare_aa_pdb(in_pdb, out_pdb)    
        
    def split_chains(self):
        parser = PDBParser()
        in_pdb = self.syspdb.replace('.pdb', '_clean.pdb')
        structure = parser.get_structure(self.sysname, in_pdb)
        io = PDBIO()
        for model in structure:
            for chain in model:
                io.set_structure(chain)
                chain_id = chain.id
                if chain.get_unpacked_list()[0].get_resname() in self.nuc_resnames:
                    out_pdb = os.path.join(self.nucdir, f'chain_{chain_id}.pdb')
                else:
                    out_pdb = os.path.join(self.prodir, f'chain_{chain_id}.pdb')
                io.save(out_pdb)
                
    def get_go_maps(self):
        from get_go import get_go
        pbds = [os.path.join(self.prodir, file) for file in os.listdir(self.prodir)]
        get_go(self.mapdir, pbds)
        
    def martinize_proteins(self):
        print("Working on proteins")
        with open(os.path.join(self.topdir, 'go_atomtypes.itp'), 'w') as file:
            file.write(f'[ atomtypes ]\n')
        with open(os.path.join(self.topdir, 'go_nbparams.itp'), 'w') as file:
            file.write(f'[ nonbond_params ]\n')
        from martini_tools import martinize_go
        for file in sorted(os.listdir(self.prodir)):
            in_pdb = os.path.join(self.prodir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            go_moltype = file.split('.')[0]
            go_map = os.path.join(self.mapdir, f'{go_moltype}.map')
            martinize_go(self.wdir, self.topdir, in_pdb, cg_pdb, go_map, go_moltype, go_eps=9.414, go_low=0.3, go_up=1.1, go_res_dist=3)
    
    def martinize_nucleotides(self):
        print("Working on nucleotides")
        from martini_tools import martinize_nucleotide
        for file in os.listdir(self.nucdir):
            in_pdb = os.path.join(self.nucdir, file)
            cg_pdb = os.path.join(self.cgdir, file)
            go_moltype = file.split('.')[0]
            martinize_nucleotide(self.wdir, self.topdir, in_pdb, cg_pdb)
        bdir = os.getcwd()
        nfiles = [f for f in os.listdir(self.wdir) if f.startswith('Nucleic')]
        for f in nfiles:
            file = os.path.join(self.wdir, f)
            command = f'sed -i s/Nucleic_/chain_/g {file}'
            sp.run(command.split())
            outfile = f.replace('Nucleic', 'chain')
            shutil.move(os.path.join(self.wdir, file), os.path.join(self.topdir, outfile))

    def make_cgpdb_file(self, d=1.25, bt='dodecahedron'):
        with open(self.syscg, 'w') as outfile:
            pass 
        with open(self.syscg, 'a') as outfile:
            for filename in sorted(os.listdir(self.cgdir)):
                file_path = os.path.join(self.cgdir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as infile:
                        for line in infile:
                            if line.startswith('ATOM'):
                                outfile.write(line)
        command = f'gmx_mpi editconf -f {self.syscg} -d {d} -bt {bt}  -o {self.syscg}'
        sp.run(command.split())
            
    def make_topology_file(self):
        itp_files = sorted([f for f in os.listdir(self.topdir) if f.endswith(".itp") and not (f.startswith("martini") or f.startswith("go_"))])
        with open(self.systop, 'w') as f:
            f.write(f'#define GO_VIRT"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_rna.itp"\n')
            f.write(f'#include "{self.topdir}/go_atomtypes.itp"\n')
            f.write(f'#include "{self.topdir}/go_nbparams.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_solvents_v1.itp"\n') 
            f.write(f'#include "{self.topdir}/martini_v3.0.0_phospholipids_v1.itp"\n')
            f.write(f'#include "{self.topdir}/martini_v3.0.0_ions_v1.itp"\n')
            f.write(f'#define RUBBER_BANDS\n')
            f.write(f'\n')
            for filename in itp_files:
                if filename.endswith('.itp'):
                    f.write(f'#include "{self.topdir}/{filename}"\n')
            f.write(f'\n[ system ]\n')
            f.write(f'Martini system for {self.sysname}\n') 
            f.write('\n[molecules]\n')
            f.write('; name\t\tnumber\n')
            for filename in itp_files:
                if filename.endswith('.itp'):
                    molecule_name = os.path.splitext(filename)[0]
                    f.write(f'{molecule_name}\t\t1\n')
            
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
        cli.solvate(self.wdir, **kwargs)
        
    def add_ions(self, system='system.pdb', conc=0.15, pname='NA', nname='CL'): 
        bdir = os.getcwd()
        os.chdir(self.wdir)
        command = f'gmx_mpi grompp -f mdp/ions.mdp -c {system} -p system.top -o ions.tpr'
        sp.run(command.split())
        command = f'gmx_mpi genion -s ions.tpr -p system.top -conc {conc} -neutral -pname {pname} -nname {nname} -o {system}'
        sp.run(command.split(), input='W\n', text=True) 
        os.chdir(bdir)
        
    def make_index_file(self, **kwargs):
        cli.make_ndx(self.wdir, **kwargs)
        
    def init_md(self, runname):
        mdrun = MDRun(runname, self.sysdir, self.sysname, self.syspdb)
        self.mdruns.append(mdrun)
        return mdrun
        
    def _get_chain_ids(self):
        chain_ids = []
        # List all files in the directory
        for filename in sorted(os.listdir(self.cgdir)):
            if filename.endswith('.pdb'):
                chain_id = filename.split('.')[0].split('_')[1]
                chain_ids.append(chain_id)
        return chain_ids
    
    @staticmethod    
    def mask_atoms(inpdb='system.pdb', outpdb='pca.pdb', atom_names=['BB', 'BB1', 'BB2', 'BB3']):
        filtered_atoms = []
        with open(inpdb, 'r') as file:
            for line in file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    current_atom_name = line[12:16].strip()
                    if current_atom_name in atom_names:
                        filtered_atoms.append(line)
        with open(outpdb, 'w') as output_file:
            for i, atom_line in enumerate(filtered_atoms, start=1):
                new_serial_number = f'{i:>5}'  
                new_atom_line = atom_line[:6] + new_serial_number + atom_line[11:]
                output_file.write(new_atom_line)
                
    @staticmethod    
    def make_index_by_chain(inpdb='pca.pdb', outndx='pca.ndx', chains=[]):
        commands = [f'chain {x}\n' for x in chains]
        clinput = 'keep 0\n' + ' '.join(commands) + 'q\n'
        cli.run('gmx_mpi make_ndx', f=inpdb, o=outndx, clinput=clinput)
        
    def get_masked_pdb_ndx(self, outpdb='pca.pdb', outndx='pca.ndx', atom_names=['BB', 'BB1', 'BB2', 'BB3']):
        bdir = os.getcwd()
        os.chdir(self.wdir)
        System.mask_atoms(inpdb=self.syscg, outpdb=outpdb, atom_names=atom_names)
        System.make_index_by_chain(chains=self.chains, outndx=outndx)
        os.chdir(bdir)    

        
        
class MDRun(System):
    
        def __init__(self, runname, *args, **kwargs):
            """
            Init
            """
            super().__init__(*args)
            self.runname = runname
            self.rundir = os.path.join(self.wdir, runname)
            self.rmsdir = os.path.join(self.rundir, 'rms_analysis')
            self.covdir = os.path.join(self.rundir, 'cov_analysis')
            self.pngdir = os.path.join(self.rundir, 'png')
            os.makedirs(self.rundir, exist_ok=True)
            os.makedirs(self.rmsdir, exist_ok=True)
            os.makedirs(self.covdir, exist_ok=True)
            os.makedirs(self.pngdir, exist_ok=True)
        
        def em(self, mdp='../mdp/em.mdp'):
            cli.mdrun(self.rundir, mdp=mdp, c='../system.pdb', r='../system.pdb', 
                p='../system.top', o='em.tpr', deffnm='em', ncpus=8)
                
        def hu(self, mdp='../mdp/hu.mdp'):
            cli.mdrun(self.rundir, mdp=mdp, c='em.gro', r='em.gro', 
                p='../system.top', o='hu.tpr', deffnm='hu', ncpus=8)
                
        def eq(self, mdp='../mdp/eq.mdp'):
            cli.mdrun(self.rundir, mdp=mdp, c='hu.gro', r='hu.gro', 
                p='../system.top', o='eq.tpr', deffnm='eq', ncpus=8)
                
        def md(self, mdp='../mdp/md.mdp'):
            cli.mdrun(self.rundir, mdp=mdp, c='eq.gro',
                p='../system.top', o='md.tpr', deffnm='md', ncpus=8)
                
        def trjconv(self, clinput=None, **kwargs):
             cli.trjconv(self.rundir, clinput=clinput, **kwargs)
             
        def rmsf(self, clinput=None, **kwargs):
             cli.rmsf(self.rundir, clinput=clinput, **kwargs)
             
        def get_rmsf_by_chain(self, **kwargs):
            kwargs.setdefault('f', 'pca.xtc')
            kwargs.setdefault('s', os.path.join(self.wdir, 'pca.pdb'))
            kwargs.setdefault('n', os.path.join(self.wdir, 'pca.ndx'))
            for chain in self.chains:
                cli.rmsf(self.rundir, clinput=f'ch{chain}\n', 
                    o=os.path.join(self.rmsdir, f'rmsf_{chain}.xvg'), **kwargs)
                    
        def get_rmsd_by_chain(self, **kwargs):
            kwargs.setdefault('f', 'pca.xtc')
            kwargs.setdefault('s', os.path.join(self.wdir, 'pca.pdb'))
            kwargs.setdefault('n', os.path.join(self.wdir, 'pca.ndx'))
            for chain in self.chains:
                cli.rms(self.rundir, clinput=f'ch{chain}\nch{chain}\n', 
                    o=os.path.join(self.rmsdir, f'rmsd_{chain}.xvg'), **kwargs)
                
            
            

        

        

    
