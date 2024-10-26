import os
import sys
sys.path.append('cgtools')
from system import CGSystem
from cli import sbatch, run

script = sys.argv[1]
sysdir = 'systems'

# sysnames = ['4zt0', '8aw3'] # 4zt0 8aw3 100bpRNA rna_test
# runs = ['mdrun_81', 'mdrun_82',] 
# 
sysnames = ['30S']
runs = ['mdrun_31']


def submit_setup(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        if submit:
            sbatch(script, 'run_all.py', 'setup', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        else:
            run('bash', script, 'run_all.py', 'setup', sysdir, sysname)


def submit_md(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'md', sysdir, sysname, runname, N=1, n=1, c=6, t='00-12:00:00', gres='gpu:1', mem='4G', qos='public', partition='general')
            else:
                run('bash', script, 'run_all.py', 'md', sysdir, sysname, runname)


def submit_extend(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'extend', sysdir, sysname, runname, N=1, n=1, c=6, t='04-00:00:00', gres='gpu:1', mem='1G', qos='public', partition='general')
            else:
                run('bash', script, 'run_all.py', 'extend', sysdir, sysname, runname)
  
            
def submit_geometry():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'geometry', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'geometry', sysdir, sysname, runname)
            run(f'python geometry_3bb.py {sysname} {runname}')


def submit_trjconv(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'trjconv', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            else:
                run('bash', script, 'run_all.py', 'trjconv', sysdir, sysname, runname)

            
def submit_rms_analysis(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'rms_analysis', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'rms_analysis', sysdir, sysname, runname)
            run('bash', script, 'run_all.py', 'plot', sysdir, sysname, runname)
            
            
def submit_plot(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'plot', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'plot', sysdir, sysname, runname)
           
            
# submit_setup(submit=False)
# submit_md()
# submit_extend()
submit_geometry()
# submit_trjconv(submit=False)
# submit_rms_analysis(submit=False)
# submit_plot_script(submit=False)