import os
import sys
sys.path.append('cgtools')
from system import CGSystem
from cli import sbatch, run

script = sys.argv[1]
sysdir = 'systems'
sysnames = ['8aw3']
# sysnames = [d for d in os.listdir(sysdir) if not d.startswith('#') and not d.startswith('test')]


def submit_setup_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        # sbatch(script, 'run_all.py', 'setup', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        run('bash', script, 'run_all.py', 'setup', sysdir, sysname)


def submit_md_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in ['mdrun_0', 'mdrun_1']:
            mdrun = system.initmd(runname)
            sbatch(script, 'run_all.py', 'md', sysdir, sysname, runname, N=1, n=1, c=6, t='04-00:00:00', gres='gpu:1', mem='2G', qos='grp_sozkan', partition='general')
            # run('bash', script, 'run_all.py', 'md', sysdir, sysname, runname)


def submit_extend_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in system.mdruns:
            mdrun = system.initmd(runname)
            sbatch(script, 'run_all.py', 'extend', sysdir, sysname, runname, N=1, n=1, c=6, t='04-00:00:00', gres='gpu:1', mem='2G', qos='grp_sozkan', partition='general')
            # run('bash', script, 'run_all.py', 'extend', sysdir, sysname, runname)
            
            
def submit_analysis_script():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in ['mdrun_0', 'mdrun_1']:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'analysis', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'analysis', sysdir, sysname, runname)
           
            
# submit_setup_script()
# submit_md_script()
submit_analysis_script()