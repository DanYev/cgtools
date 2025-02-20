import argparse
import importlib.resources
import numpy as np
import os
import pandas as pd
import shutil
import subprocess as sp
from pathlib import Path
# local
from cgtools.martini.get_go import get_go
from cgtools import cli


def append_to(in_file, out_file):
    with open(in_file, 'r') as src:
        lines = src.readlines()
    with open(out_file, 'a') as dest:
        dest.writelines(lines[1:])


def prt_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs CGMD simulation for pyRosetta")
    parser.add_argument("-f", "--pdb", required=True, help="PDB file")
    parser.add_argument("-d", "--wdir", required=True, help="Relative path to the working directory")
    parser.add_argument("-n", "--ncpus", required=False, help="Number of requested CPUS for a batch job")
    return parser.parse_args()

                    
def make_topology_file(wdir, protein='protein'):
    r"""
    -protein        Name of the protein (just for the reference, doesn't affect anything)
    
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    with open('system.top', 'w') as out_file:
        out_file.write(f'#define GO_VIRT\n')
        out_file.write(f'#include "martini.itp"\n')
        out_file.write(f'#include "go_atomtypes.itp"\n')
        out_file.write(f'#include "go_nbparams.itp"\n')
        out_file.write(f'#include "protein.itp"\n')
        out_file.write(f'#include "solvents.itp"\n')
        out_file.write(f'#include "ions.itp"\n')
        out_file.write(f'\n[ system ]\n')
        out_file.write(f'Martini protein in water\n\n') 
        out_file.write(f'\n[ molecules ]\n')
        out_file.write(f'{protein}  1\n')
    os.chdir(bdir)
        
        
def link_itps(wdir):
    bdir = os.getcwd()
    os.chdir(wdir)
    for name, path in PRT_DICT.items():
        if name.endswith('.itp'):
            command = f'ln -sf {path} {name}' # > /dev/null 2>&
            sp.run(command.split())
    os.chdir(bdir)
    
    
def gmx_pdb(wdir, in_pdb, out_pdb):
    bdir = os.getcwd()
    os.chdir(wdir)
    command = f'gmx_mpi pdb2gmx -f {in_pdb} -o clean.pdb -water none -ff amber94 -renum -ignh' # 
    sp.run(command.split())
    with open(out_pdb, 'w') as fd:
        sp.run(['grep', '^ATOM', 'clean.pdb'], stdout=fd)
    os.remove('clean.pdb')
    os.chdir(bdir)
    
    
def fix_go_map(wdir, in_map, out_map='go.map'):
    bdir = os.getcwd()
    os.chdir(wdir)
    with open (in_map, 'r') as in_file:
         with open (out_map, 'w') as out_file:
            for line in in_file:
                if line.startswith('R '):
                    new_line = ' '.join(line.split()[:-1])
                    out_file.write(f'{new_line}\n')
    os.chdir(bdir)


def prepare_files(pdb, wdir='test', mutations=None, protein='protein'):
    r"""
    -wdir           Relative path to the working directory
    """
    os.makedirs(wdir, exist_ok=True)
    copy_from = os.path.join(wdir, 'protein_minimized.pdb')
    copy_to = os.path.join(wdir, 'protein.pdb')
    shutil.copy(copy_from, copy_to)
    link_itps(wdir)
    make_topology_file(wdir, protein=protein)
    print("Getting Go-map...")
    get_go(wdir, protein)
    fix_go_map(wdir, in_map='protein_map.map')
    print('All the files are ready!')
    
@cli.from_wdir    
def martinize_go(wdir, topdir, aapdb, cgpdb, go_moltype='protein', 
    go_eps=9.414, go_low=0.3, go_up=1.1, go_res_dist=3, **kwargs):
    """
    Virtual site based GoMartini:
    -go_map         Contact map to be used for the Martini Go model.Currently, only one format is supported. (default: None)
    -go_moltype     Set the name of the molecule when using Virtual Sites GoMartini. (default: protein)
    -go_eps         The strength of the Go model structural bias in kJ/mol. (default: 9.414)                        
    -go_low         Minimum distance (nm) below which contacts are removed. (default: 0.3)
    -go_up          Maximum distance (nm) above which contacts are removed. (default: 1.1)
    -go_res_dist    Minimum graph distance (similar sequence distance) below which contacts are removed. (default: 3)
    """
    kwargs.setdefault('f', aapdb)
    kwargs.setdefault('x', cgpdb)
    kwargs.setdefault('go', 'go_map')
    kwargs.setdefault('o', 'protein.top')
    kwargs.setdefault('cys', 0.3)  
    kwargs.setdefault('p', 'all')
    kwargs.setdefault('pf', 1000)    
    kwargs.setdefault('dssp', ' ')
    kwargs.setdefault('sep', ' ')
    kwargs.setdefault('scfix', ' ')
    kwargs.setdefault('resid', 'input')
    kwargs.setdefault('ff', 'martini3001')
    kwargs.setdefault('maxwarn', '1000')
    line = f'-go-moltype {go_moltype} -go-eps {go_eps} -go-low {go_low} -go-up {go_up} -go-res-dis {go_res_dist}'
    try:
        cli.run('martinize2', line, **kwargs)
    except:
        print('Error')
    append_to('go_atomtypes.itp', os.path.join(topdir, 'go_atomtypes.itp'))
    append_to('go_nbparams.itp', os.path.join(topdir, 'go_nbparams.itp'))
    shutil.move(f'{go_moltype}.itp',  os.path.join(topdir, f'{go_moltype}.itp'))


@cli.from_wdir    
def martinize_en(wdir, aapdb, cgpdb, elastic=' ', 
    ef=1000, el=0.0, eu=0.9,  **kwargs):
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
      -eunit RB_UNIT        Establish what is the structural unit for the elastic network. Bonds are only created within a unit. Options are molecule, chain, all, or aspecified region defined by resids,
                            with followingformat: <start_resid_1>:<end_resid_1>, <start_resid_2>:<end_resid_2>... (default: molecule)
    """
    kwargs.setdefault('f', aapdb)
    kwargs.setdefault('x', cgpdb)
    kwargs.setdefault('o', 'protein.top')
    kwargs.setdefault('cys', 0.3)  
    kwargs.setdefault('p', 'all')
    kwargs.setdefault('pf', 1000)    
    kwargs.setdefault('dssp', ' ')
    kwargs.setdefault('sep', ' ')
    kwargs.setdefault('scfix', ' ')
    kwargs.setdefault('resid', 'input')
    kwargs.setdefault('ff', 'martini3001')
    kwargs.setdefault('maxwarn', '1000')
    line = f'-elastic {elastic} -ef {ef} -eu {eu}'
    try:
        cli.run('martinize2', line, **kwargs)
    except:
        print('Error')
    
    
def martinize_nucleotide(wdir, aapdb, cgpdb, **kwargs):
    kwargs.setdefault('f', aapdb)
    kwargs.setdefault('x', cgpdb)
    kwargs.setdefault('sys', 'RNA')
    kwargs.setdefault('type', 'ss')
    kwargs.setdefault('o', 'topol.top')
    kwargs.setdefault('p', 'bb')
    kwargs.setdefault('pf', 1000)
    bdir = os.getcwd()
    os.chdir(wdir)
    script = 'cgtools.martini.martinize_nucleotides'
    cli.run('python3 -m', script, **kwargs)
    os.chdir(bdir)
    

def martinize_rna(wdir, **kwargs):
    """
    Usage: python test_forge.py -f ssRNA.pdb -mol rna -elastic yes -ef 100 -el 0.5 -eu 1.2 -os molecule.pdb -ot molecule.itp
    """
    bdir = os.getcwd()
    os.chdir(wdir)
    script = 'cgtools.martini.martinize_rna'
    cli.run('python3 -m', script, **kwargs)
    os.chdir(bdir)

if __name__  == "__main__":
    pass



