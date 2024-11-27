import os
import sys
sys.path.append('cgtools')
from system import CGSystem
from cli import sbatch, run


def submit_setup(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        if submit:
            sbatch(script, 'run_all.py', 'setup', sysdir, sysname, e='slurm_output/error.%A.err', mem='10G', N=1, n=1, c=1, t='03:45:00')
        else:
            run('bash', script, 'run_all.py', 'setup', sysdir, sysname)


def submit_md(submit=True, ntomp=8):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'md', sysdir, sysname, runname, ntomp, e='slurm_output/error.%A.err', J='md',
                    t='05-00:00:00', N=1, n=1, c=ntomp, gres='gpu:1', mem='4G', 
                    qos='grp_sozkan', partition='general')
            else:
                run('bash', script, 'run_all.py', 'md', sysdir, sysname, runname)


def submit_extend(submit=True, ntomp=8):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'extend', sysdir, sysname, runname, ntomp,  e='slurm_output/error.%A.err', J='rmd',
                t='07-00:00:00', N=1, n=1, c=ntomp, gres='gpu:1', mem='3G', 
                qos='grp_sozkan', partition='general')
            else:
                run('bash', script, 'run_all.py', 'extend', sysdir, sysname, runname)
  
            
def submit_geometry():
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            # sbatch(script, 'run_all.py', 'geometry', sysdir, sysname, runname, N=1, n=1, c=1, t='00:15:00')
            run('bash', script, 'run_all.py', 'geometry', sysdir, sysname, runname)
            # run(f'python geometry_3bb.py {sysname} {runname}')


def submit_make_ndx(submit=False):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        if submit:
            sbatch(script, 'run_all.py', 'make_ndx', sysdir, sysname, N=1, n=1, c=1, mem='2G', t='00:30:00')
        else:
            run('bash', script, 'run_all.py', 'make_ndx', sysdir, sysname)
                

def submit_trjconv(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'trjconv', sysdir, sysname, runname, J='trjconv', N=1, n=1, c=1, mem='4G', t='02:00:00')
            else:
                run('bash', script, 'run_all.py', 'trjconv', sysdir, sysname, runname)

            
def submit_rms_analysis(submit=True):
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        for runname in runs:
            mdrun = system.initmd(runname)
            if submit:
                sbatch(script, 'run_all.py', 'rms_analysis', sysdir, sysname, runname, J='rms', N=1, n=1, c=1, t='03:00:00')
            else:
                run('bash', script, 'run_all.py', 'rms_analysis', sysdir, sysname, runname)
                
                
def submit_rdf_analysis(submit=True):
    for sysname in sysnames:
        for runname in runs:
            if submit:
                sbatch(script, 'run_all.py', 'rdf_analysis', sysdir, sysname, runname, J='rdf', N=1, n=1, c=1, t='03:00:00', qos='public', partition='htc')
            else:
                run('bash', script, 'run_all.py', 'rdf_analysis', sysdir, sysname, runname)


def submit_cluster(submit=True):
    for sysname in sysnames:
        for runname in runs:
            if submit:
                sbatch(script, 'run_all.py', 'cluster', sysdir, sysname, runname, 
                J='cluster', N=1, n=1, c=1, mem='2G', t='01:00:00',
                qos='public', partition='general')
            else:
                run('bash', script, 'run_all.py', 'cluster', sysdir, sysname, runname)  
                

def submit_cov_analysis(submit=True):
    for sysname in sysnames:
        for runname in runs:
            if submit:
                sbatch(script, 'run_all.py', 'cov_analysis', sysdir, sysname, runname, 
                J='cov', N=1, n=1, c=1, mem='12G', t='03:50:00',
                qos='public', partition='general')
            else:
                run('bash', script, 'run_all.py', 'cov_analysis', sysdir, sysname, runname)            


def submit_overlap(submit=False):
    for sysname in sysnames:
        if submit:
            sbatch(script, 'run_all.py', 'overlap', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        else:
            run('bash', script, 'run_all.py', 'overlap', sysdir, sysname)
 
 
def submit_get_averages(submit=False):
    for sysname in sysnames:
        if submit:
            sbatch(script, 'run_all.py', 'get_averages', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        else:
            run('bash', script, 'run_all.py', 'get_averages', sysdir, sysname) 
 
            
def submit_plot(submit=False):
    for sysname in sysnames:
        if submit:
            sbatch(script, 'run_all.py', 'plot', sysdir, sysname, N=1, n=1, c=1, t='00:15:00')
        else:
            run('bash', script, 'run_all.py', 'plot', sysdir, sysname)
            
            
def submit_test(submit=False):
    for sysname in sysnames:
        for runname in runs:
            if submit:
                sbatch(script, 'run_all.py', 'test', sysdir, sysname, runname, N=1, n=1, c=1, t='02:00:00')
            else:
                run('bash', script, 'run_all.py', 'test', sysdir, sysname, runname)
                
                
script = sys.argv[1]

sysdir = 'ribosomes' 
sysnames = ['ribosome', 'ribosome_k', ] 
runs = ['mdrun_1',  'mdrun_2', 'mdrun_3', 'mdrun_4', 'mdrun_5']  
runs += ['mdrun_6', 'mdrun_7', 'mdrun_8', 'mdrun_9', 'mdrun_10'] 
runs += ['mdrun_11', 'mdrun_12']

sysdir = 'systems' 
sysnames = ['1btl']
runs = ['mdrun_1', ]

# sysnames = ['ribosome', ] 
# runs = ['mdrun_1', ] 


# submit_setup(submit=False)
# submit_md(ntomp=8)
# submit_extend(ntomp=8)
# submit_geometry()
# submit_make_ndx(submit=False)
# submit_trjconv(submit=True)
# submit_rdf_analysis(submit=True)
# submit_cluster(submit=False)
# submit_rms_analysis(submit=True)
# submit_cov_analysis(submit=True)
# submit_overlap(submit=True)
# submit_get_averages(submit=False)
# submit_plot(submit=False)
submit_test(submit=False)

