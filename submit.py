import os
import sys
sys.path.append('cgtools')
from system import CGSystem
from cli import sbatch, run

script = sys.argv[1]
sysdir = 'systems'
sysnames = ['100bpRNA'] # 4zt0 8aw3 100bpRNA
runs = ['mdrun_51', 'mdrun_52', 'mdrun_53', 'mdrun_54'] # 
runs = ['mdrun_53']
# sysnames = [d for d in os.listdir(sysdir) if not d.startswith('#') and not d.startswith('test')]


def submit_setup_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        # sbatch(script, 'run_all.py', 'setup', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        run('bash', script, 'run_all.py', 'setup', sysdir, sysname)


def submit_md_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            sbatch(script, 'run_all.py', 'md', sysdir, sysname, runname, N=1, n=1, c=6, t='04-00:00:00', gres='gpu:1', mem='2G', qos='grp_sozkan', partition='general')
            # run('bash', script, 'run_all.py', 'md', sysdir, sysname, runname)


def submit_extend_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            sbatch(script, 'run_all.py', 'extend', sysdir, sysname, runname, N=1, n=1, c=6, t='04-00:00:00', gres='gpu:1', mem='2G', qos='public', partition='general')
            # run('bash', script, 'run_all.py', 'extend', sysdir, sysname, runname)
            
            
def submit_analysis_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'analysis', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'analysis', sysdir, sysname, runname)
            run(f'python geometry_3bb.py {sysname} {runname}')
            # run('bash', script, 'run_all.py', 'plot', sysdir, sysname, runname)
            
            
def submit_plot_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'analysis', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'plot', sysdir, sysname, runname)
           
            
submit_setup_script()
# submit_md_script()
# submit_extend_script()
# submit_analysis_script()
# submit_plot_script()