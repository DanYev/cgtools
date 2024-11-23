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
    # system.clean_inpdb(add_missing_atoms=True, add_hydrogens=True, variant=None)
    # system.split_chains(from_clean=False)
    # system.clean_proteins(add_hydrogens=True)
    # system.get_go_maps()
    # system.martinize_proteins(go_eps=9.414, go_low=0.3, go_up=0.8, p='all', pf=500, resid='mol')
    # system.martinize_nucleotides(sys='test', p='bb', pf=500, type='ss')
    # system.make_cgpdb_file(add_ions=True, bt='triclinic', box='31.0  31.0  31.0', angles='60.00  60.00  90.00')
    # system.make_cgpdb_file(bt='octahedron', d='1.25', )
    system.make_topology_file(ions=['K', 'MG', 'MGH'])
    system.solvate()
    system.add_ions(conc=0.15, pname='K', nname='CL')
    
    
def md(sysdir, sysname, runname, **kwargs): 
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.prepare_files()
    # mdrun.empp()
    # mdrun.mdrun(deffnm='em')
    # mdrun.hupp()
    # mdrun.mdrun(deffnm='hu')
    # mdrun.eqpp()
    # mdrun.mdrun(deffnm='eq')
    # mdrun.mdpp()
    mdrun.mdrun(deffnm='md', ei=os.path.join(mdrun.mdpdir, 'sam_4_mol.edi'))    
    
    
def extend(sysdir, sysname, runname, **kwargs):    
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun(deffnm='md', cpi='md.cpt', nsteps='-1', ei=os.path.join(mdrun.mdpdir, 'sam_4_mol.edi')) 


def geometry(sysdir, sysname, runname, **kwargs):  
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    mdrun.trjconv(clinput='17\n', s='mdc.pdb', f='mdc.xtc', o='chain.pdb', n=mdrun.mdcndx, pbc='nojump', dt=30000, **kwargs) # 
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
    # traj pdb and ndx
    mask = ['BB', ]
    system.make_trj_pdb_ndx(mask=mask)
    # # rdf ndx
    # system.make_ndx(pdb=system.mdcpdb, ndx='rdf.ndx', groups=[['BB1'], ['MG'], ['K'], ['CL']])
    # bb ndx
    system.make_ndx(pdb=system.mdcpdb, ndx='bb.ndx', groups=[['BB', ],] )
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    shutil.copy('atommass.dat', os.path.join(mdrun.rundir, 'atommass.dat'))
    # mdrun.trjconv(clinput='1\n', f='md.trr', o='traj.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', dt=10000, **kwargs)
    mdrun.trjconv(clinput='1\n', f='md.trr', o='mdc.pdb', n=mdrun.sysndx, pbc='atom', ur='compact', e=0)
    mdrun.trjconv(clinput='1\n', f='md.trr', o='mdc.xtc', n=mdrun.sysndx, pbc='atom', ur='compact', dt=1000, **kwargs)
    mdrun.trjconv(clinput='0\n0\n', s='mdc.pdb', f='mdc.xtc', o='mdc.xtc', pbc='nojump', **kwargs)
    mdrun.trjconv(clinput='1\n1\n1\n', s='mdc.pdb', f='mdc.pdb', o='traj.pdb', n=mdrun.bbndx, fit='rot+trans', **kwargs)
    mdrun.trjconv(clinput='1\n1\n1\n', s='mdc.pdb', f='mdc.xtc', o='traj.xtc', n=mdrun.bbndx, fit='rot+trans', **kwargs)
    # mdrun.trjconv(clinput='0\n0\n', s='traj.pdb', f='traj.xtc', o='vis.pdb', dt=15000, fit='rot+trans', **kwargs)

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 0
    e = 1000000
    mdrun.rmsf(clinput=f'0\n 0\n', f='traj.xtc', s='traj.pdb', n=system.trjndx, res='yes', b=b, e=e, )
    # mdrun.get_rmsf_by_chain(f='traj.xtc', s='traj.pdb', b=b, e=e, **kwargs)
    # mdrun.rmsf(clinput=f'1\n 1\n', b=b, n=mdrun.bbndx, res='yes')
    # mdrun.get_rmsf_by_chain(b=b, **kwargs)
    # mdrun.get_rmsd_by_chain(b=0, **kwargs)
    
    
def rdf_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    rdfndx = os.path.join(system.wdir, 'rdf.ndx')
    ions = ['MG', 'K', 'CL']
    b = 400000
    for ion in ions:
        mdrun.rdf(clinput=f'BB1\n {ion}\n', n=rdfndx, o=f'rms_analysis/rdf_{ion}.xvg', 
            b=b, rmax=10, bin=0.01, **kwargs)


def cluster(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b = 200000
    mdrun.cluster(b=b, dt=1000, cutoff=0.26, method='gromos', cl='clusters.pdb', clndx='cluster.ndx', av='yes')
    mdrun.extract_cluster()


def cov_analysis(sysdir, sysname, runname, **kwargs):
    system = CGSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    b1 = 00000
    b2 = 250000
    dt = 250000
    # f = os.path.join(mdrun.cludir, 'trajout_Cluster_0001.xtc')
    f = '../traj.xtc'
    # s = os.path.join(mdrun.cludir, 'clusters.pdb')
    s = mdrun.trjpdb
    mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=500, b=b1, e=b1+dt, ascii='covar_1.dat', o='eigenval_1.xvg', v='eigenvec_1.trr', av='av_1.pdb', l='covar_1.log')
    mdrun.covar(clinput=f'0\n 0\n', f=f, s=s, ref='yes', last=500, b=b2, e=b2+dt, ascii='covar_2.dat', o='eigenval_2.xvg', v='eigenvec_2.trr', av='av_2.pdb', l='covar_2.log')
    mdrun.anaeig(v='eigenvec_1.trr',  v2='eigenvec_2.trr', over='overlap.xvg', **kwargs) #  inpr='inprod.xpm',
    # mdrun.anaeig(clinput=f'0\n 0\n', b=b1, dt=10000, first=1, last=1, filt='filtered.pdb', v='eigenvec_1.trr', eig='eigenval_1.xvg', 
    #     comp='eigcomp.xvg', rmsf='eigrmsf.xvg', proj='proj.xvg', **kwargs) 
    # mdrun.covar(clinput=f'0\n 0\n', last=100)
    # mdrun.make_edi(clinput=f'0\n', s='../md.tpr', linfix='1-3')
    
    
def overlap(sysdir, sysname, **kwargs):
    system = CGSystem(sysdir, sysname)
    run1 = system.initmd('mdrun_4')
    run2 = system.initmd('mdrun_5')
    run3 = system.initmd('mdrun_6')
    v1 = os.path.join(run1.covdir, 'eigenvec_1.trr')
    v2 = os.path.join(run2.covdir, 'eigenvec_1.trr')
    v3 = os.path.join(run3.covdir, 'eigenvec_1.trr')
    run1.anaeig(v=v1, v2=v2, over='overlap_1.xvg', **kwargs)
    run1.anaeig(v=v2, v2=v3, over='overlap_2.xvg', **kwargs)
    run1.anaeig(v=v3, v2=v1, over='overlap_3.xvg', **kwargs)


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
    runs = system.mdruns
    files = ['eigenvec.trr', 'eigenval.xvg']
    for run in runs:
        mdrun = system.initmd(run)
        os.chdir(mdrun.rundir)
        for file in files:
            shutil.copy(file, 'cov_analysis')
        
        
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
        case 'cluster':
            cluster(*args)
        case 'cov_analysis':
            cov_analysis(*args)
        case 'overlap':
            overlap(*args)
        case 'get_averages':
            get_averages(*args)    
        case 'geometry':
            geometry(*args)
        case 'plot':
            plot_averages(*args)
        case 'test':
            test(*args)
   
        
    