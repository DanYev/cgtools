import os
import pandas as pd
import sys
import shutil
sys.path.append('cgtools')
from dataio import read_data
from system import CGSystem
import cli


def setup(sysdir, sysname):
    system = CGSystem(sysdir, sysname)
    # system.prepare_files()
    # system.clean_inpdb(add_missing_atoms=False, add_hydrogens=False, variant=None)
    # system.split_chains(from_clean=False)
    # system.get_go_maps()
    # system.martinize_proteins(go_eps=10, p='all', pf=500)
    # system.martinize_nucleotides(sys='test', p='bb', pf=1000, type='ss')
    # system.make_cgpdb_file(add_ions=True, bt='triclinic', box='31.0  31.0  31.0', angles='60.00  60.00  90.00')
    system.make_topology_file(ions=['K', 'MG',])
    # system.solvate()
    # system.add_ions(conc=0.15, pname='K', nname='CL')
    # system.get_pca_pdb_ndx()
    
    
def md(sysdir, sysname, runname, **kwargs): 
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.prepare_files()
    # mdrun.em(grompp=False, f=os.path.join(system.mdpdir, 'em0.mdp'), o='em0.tpr', deffnm='em0')
    mdrun.em()
    mdrun.hu()
    mdrun.eq() # c='em.gro', r='em.gro'
    mdrun.md(grompp=True)
    
    
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
    mdrun.trjconv(clinput=f'{trjstr}\n{trjstr}\n{trjstr}\n', f='md.trr', s='md.tpr', o='trj.pdb', n=mdrun.sysndx, pbc='whole', center=' ', ur='compact', b=50000, dt=5000) 
    # mdrun.trjconv(clinput=f'{group}\n{group}\n', f='trj.xtc', s='md.tpr', o='trj.pdb', n=mdrun.sysndx, fit='rot+trans', dt=0)


def make_ndx(sysdir, sysname, **kwargs):
    system = CGSystem(sysdir, sysname)
    atoms = ['BB', 'BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6']
    ions = ['K', 'CL', 'MG']
    # sys ndx
    system.make_ndx(pdb=system.syspdb, ndx=system.sysndx, groups=[atoms + ions])
    # mdc pdb and ndx
    mask = atoms + ions
    system.make_mdc_pdb_ndx(mask=mask)
    # rdf ndx
    system.make_ndx(pdb=system.mdcpdb, ndx='rdf.ndx', groups=[['BB1'], ['MG'], ['K'], ['CL']])
    # bb ndx
    system.make_ndx(pdb=system.mdcpdb, ndx='bb.ndx', groups=[['BB', 'BB2'],] )
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    shutil.copy('atommass.dat', os.path.join(mdrun.rundir, 'atommass.dat'))
    # mdrun.trjconv(clinput='1\n', f='md.trr', o='traj.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', dt=10000, **kwargs)
    mdrun.trjconv(clinput='1\n', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', e=0)
    mdrun.trjconv(clinput='1\n', f='md.trr', o='mdc.xtc', n=mdrun.sysndx, pbc='atom', ur='compact', dt=1500, **kwargs)

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    bbndx = os.path.join(system.wdir, 'bb.ndx')
    mdrun.rmsf(clinput=f'1\n 1\n', n=bbndx, res='yes')
    # mdrun.get_rmsf_by_chain(b=0, **kwargs)
    # mdrun.get_rmsd_by_chain(b=0, **kwargs)
    
    
def rdf_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    rdfndx = os.path.join(system.wdir, 'rdf.ndx')
    ions = ['MG', 'K', 'CL']
    for ion in ions:
        mdrun.rdf(clinput=f'BB1\n {ion}\n', n=rdfndx, o=f'rms_analysis/rdf_{ion}.xvg', 
            b=10000, rmax=10, bin=0.01, **kwargs)


def cov_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    bbndx = os.path.join(system.wdir, 'bb.ndx')
    # mdrun.covar(clinput=f'1\n 1\n', n=bbndx, last=1, **kwargs)  
    mdrun.pertmat()


def get_averages(sysdir, sysname):
    system = CGSystem(sysdir, sysname)  
    files = os.listdir(system.initmd('mdrun_1').rmsdir)
    for file in files:
        system.get_mean_sem('rms_analysis', file)
        

def plot_averages(sysdir, sysname, **kwargs):    
    from plotting import plot_each_mean_sem
    system = CGSystem(sysdir, sysname)  
    for metric in ['rdf', ]:
        fpaths = [os.path.join(system.datdir, f) for f in os.listdir(system.datdir) if f.startswith(metric)]
        plot_each_mean_sem(fpaths, system)

            
def test(sysdir, sysname):
    system = CGSystem(sysdir, sysname)
    mddir = os.path.join(system.wdir, 'mdruns')
    os.chdir(mddir)
    for idx in range(1, 6):
        odir = f'mdrun_{idx+5}'
        ndir = f'mdrun_{idx}'
        os.makedirs(ndir, exist_ok=True)
        file = 'md.tpr'
        opath = os.path.join(odir, file)
        npath = os.path.join(ndir, file)
        # print(opath, npath)
        shutil.copy(opath, ndir)
        
        
if __name__ == '__main__':
    todo = sys.argv[1]
    args = sys.argv[2:]
    match todo:
        case 'setup':
            setup(*args)
        case 'md':
            md(*args)    
        case 'extend':
            extend(*args, nsteps=-2)
        case 'make_ndx':
            make_ndx(*args)
        case 'trjconv':
            trjconv(*args)
        case 'rms_analysis':
            rms_analysis(*args)
        case 'rdf_analysis':
            rdf_analysis(*args)
        case 'cov_analysis':
            cov_analysis(*args)
        case 'get_averages':
            get_averages(*args)    
        case 'geometry':
            geometry(*args)
        case 'plot':
            plot_averages(*args)
        case 'test':
            test(*args)
   
        
    