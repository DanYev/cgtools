import os
import shutil
import sys
sys.path.append('cgtools')
from dataio import read_data
from system import CGSystem
import cli

sysdir = 'blac'
pdbsdir = 'blac/mutants_1'


def rename_mutants():
    oldnames = os.listdir(pdbsdir)
    fnames = [f.replace('cA', '') for f in oldnames]
    newnames = [f.split('_')[-1] for f in fnames]
    os.chdir(fdir)
    for of, nf in zip(oldnames, newnames):
        os.rename(of, nf)
        
        
def get_maps():
    fnames = os.listdir(pdbsdir)
    files = [os.path.join(pdbsdir, f) for f in fnames]
    mapdir = os.path.join(sysdir, 'maps')
    os.makedirs(mapdir, exist_ok=True)
    print('Getting GO-maps', file=sys.stderr)
    from get_go import get_go
    map_names = [f.replace('pdb', 'map') for f in fnames]
    # Filter out existing maps
    pdbs = [f for f, amap in zip(files, map_names) if amap not in os.listdir(mapdir)]
    pdbs = [os.path.abspath(f) for f in pdbs]
    if pdbs:
        get_go(mapdir, pdbs)
    else:
        print('Maps already there', file=sys.stderr)
        
        
def init_systems():
    fnames = os.listdir(pdbsdir)
    files = [os.path.join(pdbsdir, f) for f in fnames]
    variants = [f.split('.')[0] for f in fnames]
    dirnames = [os.path.join(sysdir, 'systems', variant) for variant in variants]
    for adir, f in zip(dirnames, files):
        os.makedirs(adir, exist_ok=True)
        shutil.copy(f, os.path.join(adir, 'inpdb.pdb'))
        

def setup_systems(sysdir='blac/systems'):
    sysnames = os.listdir(sysdir)
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        system.prepare_files()
        system.clean_inpdb(add_missing_atoms=True, add_hydrogens=False)
        system.split_chains(from_clean=True)
        # system.get_go_maps()
        # system.martinize_proteins(go_eps=9.414, go_low=0.3, go_up=0.8, p='all', pf=500, resid='mol')
        # system.martinize_nucleotides(sys='test', p='bb', pf=500, type='ss')
        # system.make_cgpdb_file(add_ions=True, bt='triclinic', box='31.0  31.0  31.0', angles='60.00  60.00  90.00')
        # system.make_topology_file(ions=['K', 'MG', 'MGH'])
        # system.solvate()
        # system.add_ions(conc=0.15, pname='K', nname='CL')      

if __name__ == '__main__':
    # init_systems()
    setup_systems()
    