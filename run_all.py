import os
import sys
import argparse
sys.path.append('cgtools')
from system import CGSystem
import cli


def parse_cmd():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs CGMD simulations")
    parser.add_argument("-m", "--mode", required=True, help="PDB file")
    return parser.parse_args()


def setup(sysdir, sysname):
    system = CGSystem(sysdir, sysname)
    # system.prepare_files()
    # system.clean_inpdb(add_missing_atoms=True, add_hydrogens=False, variant=None)
    # system.split_chains(from_clean=False)
    # # system.get_go_maps()
    # # system.martinize_proteins(go_eps=10, p='bb', pf=500)
    system.martinize_nucleotides(sys='test', p='bb', pf=1000)
    # system.make_box(d=1.25, bt='triclinic', add_ions=True)
    # system.make_topology_file()
    # system.solvate()
    # system.add_ions(conc=0.15, pname='K', nname='CL')
    # system.get_masked_pdb_ndx()
    
    
def md(sysdir, sysname, runname, **kwargs): 
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.prepare_files()
    mdrun.em()
    mdrun.hu()
    mdrun.eq() # c='em.gro', r='em.gro'
    mdrun.md()
    
    
def extend(sysdir, sysname, runname, **kwargs):    
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.extend(**kwargs)


def geometry(sysdir, sysname, runname, **kwargs):  
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    # # Protein_RNA group
    # system.make_index_file(clinput='0\nq\n', f=mdrun.syspdb, o=mdrun.sysndx)
    # group = 'Protein_RNA'
    group = 'RNA'
    atoms = ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6']
    ndxstr = 'a ' + ' | a '.join(atoms) + '\n q \n'
    trjstr = '_'.join(atoms) + '\n'
    system.make_index_file(clinput=ndxstr, f=mdrun.syspdb, o=mdrun.sysndx)
    mdrun.trjconv(clinput=f'{trjstr}\n{trjstr}\n{trjstr}\n', f='md.trr', s='md.tpr', o='trj.pdb', n=mdrun.sysndx, pbc='whole', center=' ', ur='compact', dt=1000) 
    # mdrun.trjconv(clinput=f'{group}\n{group}\n', f='trj.xtc', s='md.tpr', o='trj.pdb', n=mdrun.sysndx, fit='rot+trans', dt=0)
    

def trjconv(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    # Ugly but needed to use the index groups
    atoms = ['BB1', 'BB2', 'BB3']
    ndxstr = 'a ' + ' | a '.join(atoms) + '\n q \n'
    trjstr = '_'.join(atoms) + '\n'
    system.make_index_file(clinput=ndxstr, f=mdrun.syspdb, o=mdrun.sysndx)
    mdrun.trjconv(clinput=trjstr, f='md.trr', o='pca.xtc', n=mdrun.sysndx, pbc='nojump', ur='compact')
    mdrun.trjconv(clinput=trjstr, f='md.trr', o='pca.pdb', n=mdrun.sysndx, pbc='nojump', ur='compact', e=0)


def rms_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    # Ugly but needed to use the index groups
    atoms = ['BB1', 'BB2', 'BB3']
    trjstr = '_'.join(atoms) + '\n'
    mdrun.get_rdf(b=0)
    # mdrun.get_rmsf_by_chain(b=0)
    # mdrun.get_rmsd_by_chain(b=0)
    
    
def plot(sysdir, sysname, runname, **kwargs): 
    from plotting import Plot2D, read_data
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    os.chdir(mdrun.rundir)
    rmsf_files = [os.path.join(mdrun.rmsdir, f) for f in os.listdir(mdrun.rmsdir) if f.startswith('rmsf')]
    for file in rmsf_files:
        data = read_data(file)
        label = file.split('/')[-1]
        figname=file.replace('xvg', 'png').replace('rms_analysis', 'png')
        plot = Plot2D([[data]], [[label]], ylabel='RMSF, nm',  legend=True, loc='upper right', figname=figname)
        plot.make_plot()
        
    rmsd_files = [os.path.join(mdrun.rmsdir, f) for f in os.listdir(mdrun.rmsdir) if f.startswith('rmsd')]
    for file in rmsd_files:
        data = read_data(file)
        label = file.split('/')[-1]
        figname=file.replace('xvg', 'png').replace('rms_analysis', 'png')
        plot = Plot2D([[data]], [[label]], ylabel='RMSD, nm', xlabel='Time, ps', legend=True, loc='upper right', figname=figname)
        plot.make_plot()
            
           
def test_script(sysdir, sysname, runname):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.extend()
    
    
if __name__ == '__main__':
    todo = sys.argv[1]
    args = sys.argv[2:]
    match todo:
        case 'setup':
            setup(*args)
        case 'md':
            md(*args)    
        case 'extend':
            extend(*args, nsteps=-1)
        case 'trjconv':
            trjconv(*args)
        case 'rms_analysis':
            rms_analysis(*args)
        case 'geometry':
            rms_analysis(*args)
        case 'plot':
            plot(*args)
   
        
    