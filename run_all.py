import os
import pandas as pd
import sys
import shutil
sys.path.append('cgtools')
from dataio import read_data
from system import CGSystem
import cli
import MDAnalysis as mda
from pathlib import Path


def setup(sysdir, sysname):
    system = CGSystem(sysdir, sysname)
    # system.prepare_files()
    # system.clean_inpdb(add_missing_atoms=True, add_hydrogens=True, variant=None)
    # system.split_chains(from_clean=False)
    # system.clean_proteins(add_hydrogens=True)
    # system.get_go_maps()
    system.martinize_proteins(go_eps=10.0, go_low=0.3, go_up=1.1, p='backbone', pf=500, resid='mol')
    # system.martinize_nucleotides(sys='test', p='bb', pf=500, type='ss')
    # system.make_cgpdb_file(add_ions=True, bt='triclinic', box='31.0  31.0  31.0', angles='60.00  60.00  90.00')
    system.make_cgpdb_file(bt='octahedron', d='1.25', )
    system.make_topology_file(ions=['K', 'MG', 'MGH'])
    system.solvate()
    system.add_ions(conc=0.0, pname='K', nname='CL')
    
    
def md(sysdir, sysname, runname, ntomp): 
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.prepare_files()
    # n = os.path.join(system.wdir, 'index.ndx')
    # em
    mdrun.empp()
    mdrun.mdrun(deffnm='em', ntomp=ntomp)
    # hu
    mdrun.hupp()
    mdrun.mdrun(deffnm='hu', ntomp=ntomp)
    # eq
    mdrun.eqpp() # n=n, c='em.gro', r='em.gro'
    mdrun.mdrun(deffnm='eq', ntomp=ntomp)
    # md
    mdrun.mdpp()
    mdrun.mdrun(deffnm='md', ntomp=ntomp)    # , ei='sam.edi'
    
    
def extend(sysdir, sysname, runname, ntomp):    
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.mdrun(deffnm='ext', ntomp=ntomp, nsteps=-2)  # cpi='md.cpt'


def geometry(sysdir, sysname, runname, **kwargs):  
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.trjconv(s='mdc.pdb', f='mdc.xtc', o='chain.pdb', n=mdrun.mdcndx, pbc='nojump', dt=30000, **kwargs) # clinput='17\n', 
    # mdrun.trjconv(clinput=f'{group}\n{group}\n', f='trj.xtc', s='md.tpr', o='trj.pdb', n=mdrun.sysndx, fit='rot+trans', dt=0)


def make_ndx(sysdir, sysname, **kwargs):
    system = CGSystem(sysdir, sysname)
    atoms = ['BB', 'BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', ]
    ions = ['K', 'CL', 'MG']
    # sys ndx
    system.make_ndx(pdb=system.syspdb, ndx=system.sysndx, groups=[atoms + ions])
    # mdc pdb and ndx
    mask = atoms + ions
    system.make_mdc_pdb_ndx(mask=mask)
    # traj pdb and ndx
    backbone = ['BB', 'BB2']
    system.make_trj_pdb_ndx(mask=backbone)
    # bb ndx
    system.make_ndx(pdb=system.mdcpdb, ndx='bb.ndx', groups=[backbone])
    # # rdf ndx
    # system.make_ndx(pdb=system.mdcpdb, ndx='rdf.ndx', groups=[['BB1'], ['MG'], ['K'], ['CL']])
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    # shutil.copy('atommass.dat', os.path.join(mdrun.rundir, 'atommass.dat'))
    # mdrun.trjconv(clinput='1\n', s='md.tpr', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', e=0)
    # mdrun.trjconv(clinput='1\n', s='md.tpr', f='md.trr', o='mdc.xtc', n=mdrun.sysndx, pbc='atom', ur='compact', dt=1500, **kwargs)
    # mdrun.trjconv(clinput='0\n', f='mdc.xtc', o='mdc.xtc', pbc='nojump', **kwargs)
    # mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.pdb', o='traj.pdb', n=mdrun.bbndx, fit='rot+trans', e=0, **kwargs)
    # mdrun.trjconv(clinput='1\n1\n', s='mdc.pdb', f='mdc.xtc', o='traj.xtc', n=mdrun.bbndx, fit='rot+trans', **kwargs)
    # mdrun.trjconv(clinput='0\n0\n', s='traj.pdb', f='traj.xtc', o='vis.pdb', dt=30000, fit='rot+trans', **kwargs)


def rdf_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    rdfndx = os.path.join(system.wdir, 'rdf.ndx')
    ions = ['MG', 'K', 'CL']
    b = 400000
    for ion in ions:
        mdrun.rdf(clinput=f'BB1\n {ion}\n', n=rdfndx, o=f'rms_analysis/rdf_{ion}.xvg', 
            b=b, rmax=10, bin=0.01, **kwargs)

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 300000
    e = 10000000
    # f = 'clusters/trajout_Cluster_0001.xtc'
    f = 'traj.xtc'
    s = 'traj.pdb'
    mdrun.rmsf(clinput=f'0\n 0\n', f=f, s=s, b=b, e=e, n=system.trjndx, res='yes', fit='yes',  )
    mdrun.get_rmsf_by_chain(b=b, e=e, **kwargs)
    # mdrun.rmsf(clinput=f'1\n 1\n', b=b, n=mdrun.bbndx, res='yes')
    # mdrun.get_rmsf_by_chain(b=b, **kwargs)
    # mdrun.get_rmsd_by_chain(b=0, **kwargs)
    

def cluster(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 400000
    mdrun.cluster(b=b, dt=1000, cutoff=0.15, method='gromos', cl='clusters.pdb', clndx='cluster.ndx', av='yes')
    mdrun.extract_cluster()


def cov_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    # f = os.path.join(mdrun.cludir, 'trajout_Cluster_0001.xtc')
    # s = os.path.join(mdrun.cludir, 'clusters.pdb')
    f = '../traj.xtc'
    s = mdrun.trjpdb
    b = 000000
    mdrun.covar(clinput=f'0\n 0\n', b=b, f=f, s=s, ref='no', last=1000, ascii='covar.dat', o='eigenval.xvg', v='eigenvec.trr', av='av.pdb', l='covar.log')
    mdrun.anaeig(clinput=f'0\n 0\n', b=b, dt=15000, first=1, last=10, filt='filtered.pdb', v='eigenvec.trr', eig='eigenval.xvg', proj='proj.xvg') 
    b1 = 000000
    b2 = 150000
    dt = 150000
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b1, e=b1+dt, ascii='covar_1.dat', o='eigenval_1.xvg', v='eigenvec_1.trr', av='av_1.pdb', l='covar_1.log')
    # mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=10, b=b2, e=b2+dt, ascii='covar_2.dat', o='eigenval_2.xvg', v='eigenvec_2.trr', av='av_2.pdb', l='covar_2.log')
    # mdrun.anaeig(v='eigenvec_1.trr',  v2='eigenvec_2.trr', over='overlap.xvg', **kwargs) #  inpr='inprod.xpm',
    # mdrun.covar(clinput=f'0\n 0\n', last=100)
    # mdrun.make_edi(clinput=f'1\n', s='../md.tpr', n=system.bbndx, radacc='1-3', slope=0.01, outfrq=10000, o='../sam.edi')
    
    
def overlap(sysdir, sysname, **kwargs):
    system = CGSystem(sysdir, sysname)
    run1 = system.initmd('mdrun_2')
    run2 = system.initmd('mdrun_4')
    run3 = system.initmd('mdrun_5')
    v1 = os.path.join(run1.covdir, 'eigenvec.trr')
    v2 = os.path.join(run2.covdir, 'eigenvec.trr')
    v3 = os.path.join(run3.covdir, 'eigenvec.trr')
    run1.anaeig(v=v1, v2=v2, over='overlap_1.xvg', **kwargs)
    run1.anaeig(v=v2, v2=v3, over='overlap_2.xvg', **kwargs)
    run1.anaeig(v=v3, v2=v1, over='overlap_3.xvg', **kwargs)


def dci_dfi(sysdir, sysname, runname):
    system = CGSystem(sysdir, sysname)
    run = system.initmd(runname)
    run.prepare_files()
    run.get_covmats(b=400000, n=10)
    run.get_pertmats()
    run.get_dfi()
    run.get_dci()


def get_averages(sysdir, sysname):
    system = CGSystem(sysdir, sysname)  

    def fname_filter(f, sw='', cont='', ew=''):  
        """
        Filters a file name based on its start, substring, and end patterns.
        """
        return f.startswith(sw) and cont in f and f.endswith(ew)
        
    def filter_files(fpaths, sw='', cont='', ew=''):  
        """
        Filters files in a list using the above filter
        """
        files = [f for f in fpaths if fname_filter(f.name, sw=sw, cont=cont, ew=ew)]
        return files
        
    def pull_all_files(directory):
        """
        Recursively lists all files in the given directory and its subdirectories.
    
        Parameters:
            directory (str or Path): The root directory to start searching for files.
    
        Returns:
            list[Path]: A list of Path objects, each representing the absolute path
                        to a file within the directory and its subdirectories.
        """
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                all_files.append(Path(os.path.join(root, file)))
        return all_files
            
    def pull_files(directory, sw='', cont='', ew=''):
        """
        Pulls files all files from a directory that pass the above filter
        """
        fpaths = list_files_in_subdirectories(directory)
        files = filter_files(fpaths, sw=sw, cont=cont, ew=ew)
        return files
    
    print('starting')    
    all_files = pull_all_files(system.mddir)
    # RMSF
    print(f'Processing DFI', file=sys.stderr )
    files = filter_files(all_files, sw='rmsf.', ew='.xvg')
    system.get_mean_sem(files, f'dfi.csv', col=1)
    # DFI
    print(f'Processing DFI', file=sys.stderr )
    files = filter_files(all_files, sw='dfi', ew='.xvg')
    system.get_mean_sem(files, f'dfi.csv', col=1)
    # Chain RMSF
    # for chain in system.chains:
    #     print(f'Processing chain {chain}', file=sys.stderr )
    #     sw = f'rmsf_{chain}'
    #     files = filter_files(all_files, sw=sw, ew='.xvg')
    #     system.get_mean_sem(files, f'{sw}.csv', col=1)
    # # Chain DCI 
    # for chain in system.chains:
    #     print(f'Processing chain {chain}', file=sys.stderr )
    #     sw = f'dci_{chain}'
    #     files = filter_files(all_files, sw=sw, ew='.xvg')
    #     system.get_mean_sem(files, f'{sw}.csv', col=1)
        
    
def plot_averages(sysdir, sysname, **kwargs):    
    from plotting import plot_each_mean_sem
    system = CGSystem(sysdir, sysname)  
    for metric in ['rdf', ]:
        fpaths = [os.path.join(system.datdir, f) for f in os.listdir(system.datdir) if f.startswith(metric)]
        plot_each_mean_sem(fpaths, system)
        
        
def working(sysdir, sysname, runname):
    system = CGSystem(sysdir, sysname)
    run = system.initmd(runname)
    # cli.run_gmx(run.rundir, 'convert-trj', f='md.trr', o='md_old.trr', e=1600, tu='ns')
    # cli.gmx_grompp(run.rundir, c=system.sysgro, p=system.systop, f=os.path.join(system.mdpdir, 'md.mdp'), t='md_old.trr',  o='ext.tpr')
    # cli.gmx_mdrun(run.rundir, s='ext.tpr', deffnm='ext')
    # cli.run_gmx(run.rundir, 'trjcat', clinput='c\nc\n', cltext=True, f='md_old.trr ext.trr', o='md.trr', settime='yes')
    system.make_ref_pdb()

        
if __name__ == '__main__':
    todo = sys.argv[1]
    args = sys.argv[2:]
    match todo:
        case 'setup':
            setup(*args)
        case 'md':
            md(*args)    
        case 'extend':
            extend(*args)
        case 'make_ndx':
            make_ndx(*args)
        case 'trjconv':
            trjconv(*args)
        case 'rms_analysis':
            rms_analysis(*args)
        case 'rdf_analysis':
            rdf_analysis(*args)
        case 'cluster':
            cluster(*args)
        case 'cov_analysis':
            cov_analysis(*args)
        case 'overlap':
            overlap(*args)
        case 'dci_dfi':
            dci_dfi(*args)
        case 'get_averages':
            get_averages(*args)    
        case 'geometry':
            geometry(*args)
        case 'plot':
            plot_averages(*args)
        case 'working':
            working(*args)
   
        
    