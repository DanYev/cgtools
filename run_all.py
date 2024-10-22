import os
import sys
sys.path.append('cgtools')
from system import System

pdb = '4zt0' # 8aw3  4zt0 100bpRNA
system = System('systems', f'{pdb}_test', f'{pdb}.pdb') 
runs = ['mdrun_0', ] # 'mdrun_1'

def parse_cmd():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs CGMD simulations")
    parser.add_argument("-m", "--mode", required=True, help="PDB file")
    parser.add_argument("-d", "--wdir", required=False, help="Relative path to the working directory")
    parser.add_argument("-n", "--ncpus", required=False, help="Number of requested CPUS for a batch job")
    return parser.parse_args()


def prep_system():
    system.prepare_files()
    system.clean_pdb()
    system.split_chains()
    # system.get_go_maps()
    # system.martinize_proteins()
    system.martinize_nucleotides(system='test')
    system.make_topology_file()
    system.make_cgpdb_file()
    system.solvate()
    system.add_ions()
    system.get_masked_pdb_ndx()
    
    
def run_md(): 
    runname = runnames[0]
    mdrun = system.init_md(runname)
    mdrun.em(ncpus=6)
    mdrun.hu(ncpus=6)
    mdrun.eq(ncpus=6) # c='em.gro', r='em.gro', 
    mdrun.md(ncpus=6)
    
    
def extend():    
    runname = runnames[0]
    mdrun = system.init_md(runname)
    mdrun.extend()


def analysis():    
    for run in runs:
        mdrun = system.init_md(run)
        atoms = ['BB1', 'BB2']
        ndxstr = 'a ' + ' | a '.join(atoms) + '\n q \n'
        trjstr = '_'.join(atoms) + '\n'
        system.make_index_file(clinput=ndxstr, f='system.pdb', o=mdrun.sysndx)
        mdrun.trjconv(clinput=trjstr, f='md.trr', o='trj.pdb', n=mdrun.sysndx, pbc='nojump', ur='compact', dt=1000)
        exit()
        mdrun.trjconv(clinput=trjstr, f='md.trr', o='pca.xtc', n=mdrun.sysndx, pbc='nojump', ur='compact')
        mdrun.trjconv(clinput=trjstr, f='md.trr', o='pca.pdb', n=mdrun.sysndx, pbc='nojump', ur='compact', e=0)
        mdrun.get_rmsf_by_chain()
        mdrun.get_rmsd_by_chain()
    
    
def plot():  
    from plotting import Plot2D, read_data
    for run in runs:
        mdrun = system.init_md(run)
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
            
            
def execute_script():
    pass
    
    
def submit_scripts():
    pdbs = ['4zt0', '8aw3']
    for pdb in pdbs:
        system = System('systems', f'{pdb}_test', f'{pdb}.pdb') 
        # mdruns = [f'mdrun_{x}' for x in range(10)]
        # system.mdruns = mdruns
        # for mdrun in self.mdruns
        
    
    
    
if __name__ == '__main__':
    mode = sys.argv[1]
    match mode:
        case 'sub':
            print('Submitting scripts')
            submit_scripts()
        case 'exe':
            print('Executing the script')
        case _:
            print("Error: the second argument needs to be either 'sub' or 'exe'")
            
            
    # parse_cmd()
    # prep_system()
    # run_md()
    # analysis()
    # plot()
    # extend()
    