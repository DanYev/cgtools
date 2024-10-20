import os
from plotting import Plot2D, read_data
from system import System

pdb = '4zt0'
system = System('systems', f'test_{pdb}', f'{pdb}.pdb') 

def prep_system():
    # system.prepare_files()
    # system.clean_pdb()
    # system.split_chains()
    # system.get_go_maps()
    # system.martinize_proteins()
    # system.martinize_nucleotides()
    system.make_topology_file()
    system.make_cgpdb_file()
    system.solvate()
    system.add_ions()
    
def run_md():    
    mdrun = system.init_md('mdrun_test')
    mdrun.em()
    mdrun.hu()
    mdrun.eq()
    mdrun.md()


def analysis():    
    mdrun = system.init_md('mdrun_test')
    # system.make_index_file(clinput='a BB | a BB1 | a BB2 | a BB3 \n q \n', f='system.pdb', n='index.ndx', o='index.ndx')
    # system.get_masked_pdb_ndx()
    # mdrun.trjconv(clinput='BB_BB1_BB2_BB3 \n', f='md.trr', o='pca.xtc', n='../index.ndx', pbc='whole', ur='compact') # clinput='17\n17\n', 
    # mdrun.trjconv(clinput='BB_BB1_BB2_BB3 \n', f='md.trr', o='pca.pdb', n='../index.ndx', e=0)
    mdrun.get_rmsf_by_chain()
    mdrun.get_rmsd_by_chain()
    
    
def plot():    
    mdrun = system.init_md('mdrun_test')
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
        plot = Plot2D([[data]], [[label]], ylabel='RMSD, nm',  legend=True, loc='upper right', figname=figname)
        plot.make_plot()
    
    
if __name__ == '__main__':
    # prep_system()
    # run_md()
    # analysis()
    plot()
    