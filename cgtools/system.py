import os
import sys
import numpy as np
import shutil
import subprocess as sp
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO, Atom
import gmx_wraps


class System:
    """
    Class to set up and analyze protein-nucliotide-lipid systems
    """    
    
    nuc_resnames = ['A', 'C', 'G', 'U', 'RA3', 'RA5', 'RC3', 'RC5', 'RG3', 'RG5', 'RU3', 'RU5']
    
    def __init__(self, sysname, pdb, **kwargs):
        """
        Init
        """
        self.sysname = sysname
        self.wdir = os.path.abspath(sysname)
        self.syspdb = os.path.join(self.wdir, pdb)
        self.prodir = os.path.join(self.wdir, 'proteins')
        self.nucdir = os.path.join(self.wdir, 'nucleotides')
        self.topdir = os.path.join(self.wdir, 'topol')
        self.mapdir = os.path.join(self.wdir, 'map')
        self.mdpdir = os.path.join(self.wdir, 'mdp')
        self.cgdir  = os.path.join(self.wdir, 'cgpdb')
        self.grodir = os.path.join(self.wdir, 'gro')
        self.mdruns = []
        
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
        for file in os.listdir(self.prodir):
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
            
    def make_topology_file(self):
        itp_files = sorted([f for f in os.listdir(self.topdir) if f.endswith(".itp") and not (f.startswith("martini") or f.startswith("go_"))])
        with open(os.path.join(self.wdir, 'system.top'), 'w') as f:
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
            
    def make_gro_file(self, d=1.5, bt='dodecahedron'):
        cg_pdb_files = sorted(os.listdir(self.cgdir))
        for file in cg_pdb_files:
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
        sys_gro = os.path.join(self.wdir, 'system.gro')
        with open(sys_gro, 'w') as out_f:
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
        command = f'gmx_mpi editconf -f {sys_gro} -d {d} -bt {bt}  -o {sys_gro}'
        sp.run(command.split())
        
    def solvate(self, radius=0.23):
        bdir = os.getcwd()
        os.chdir(self.wdir)
        command = f'gmx_mpi solvate -cp system.gro -cs mdp/water.gro -p system.top -radius {radius} -o system.gro'
        sp.run(command.split())
        os.chdir(bdir)
        
    def add_ions(self, conc=0.15, pname='NA', nname='CL'): 
        bdir = os.getcwd()
        os.chdir(self.wdir)
        command = f'gmx_mpi grompp -f mdp/ions.mdp -c system.gro -p system.top -o ions.tpr'
        sp.run(command.split())
        command = f'gmx_mpi genion -s ions.tpr -p system.top -conc {conc} -neutral -pname {pname} -nname {nname} -o system.gro'
        sp.run(command.split(), input='W\n', text=True) 
        os.chdir(bdir)
        
    def init_md(self, runname):
        mdrun = MDRun(runname, self.sysname, self.syspdb)
        self.mdruns.append(mdrun)
        return mdrun
        
        
class MDRun(System):
    
        def __init__(self, runname, *args, **kwargs):
            """
            Init
            """
            super().__init__(*args)
            self.runname = runname

        

        

    
