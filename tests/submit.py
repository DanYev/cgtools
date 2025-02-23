import os
import sys
from cgtools.gmxmd import gmxSystem
from cgtools.cli import sbatch, run


def dojob(submit, *args, **kwargs):
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)

def setup_cg_model(submit=False, **kwargs): 
    kwargs.setdefault('mem', '3G')
    for sysname in sysnames:
        dojob(submit, script, pyscript, 'setup_cg_model', sysdir, sysname, 
            J=f'setup_{sysname}', **kwargs)


def md(submit=True, ntomp=8, **kwargs): #   qos='public', partition='htc',   #  qos='grp_sozkan', partition='general'
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    kwargs.setdefault('G', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'md', sysdir, sysname, runname, ntomp, 
                    J=f'md_{sysname}_{runname}', **kwargs)


def extend(submit=True, ntomp=8, **kwargs):
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    kwargs.setdefault('G', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'extend', sysdir, sysname, runname, ntomp, 
                    J=f'ext_{sysname}_{runname}', **kwargs)
                

def trjconv(submit=True, **kwargs):
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'trjconv', sysdir, sysname, runname,
                    J=f'trjconv_{sysname}_{runname}', **kwargs)

            
def rms_analysis(submit=True, **kwargs):
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'rms_analysis', sysdir, sysname, runname,
                    J=f'rms_{sysname}_{runname}', **kwargs)
                
            
def cov_analysis(submit=True, **kwargs):
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'cov_analysis', sysdir, sysname, runname,
                    J=f'cov_{sysname}_{runname}', **kwargs)
      

 
def dci_dfi(submit=True):
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'dci_dfi', sysdir, sysname, runname,
                    J=f'dci_dfi_{sysname}_{runname}', **kwargs)


def tdlrt(submit=True, **kwargs):
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '30G')
    kwargs.setdefault('G', '1')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'tdlrt', sysdir, sysname, runname,
                    J=f'tdlrt_{sysname}_{runname}', **kwargs)


def tdlrt_figs(submit=True, **kwargs):
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'tdlrt_figs', sysdir, sysname, runname,
                    J=f'plot_{sysname}_{runname}', **kwargs)
 

def get_averages(submit=False, **kwargs):
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        dojob(submit, script, pyscript, 'get_averages', sysdir, sysname, 
            J=f'av_{sysname}', **kwargs)


def test(submit=False, **kwargs):
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'test', sysdir, sysname, runname,
                    J=f'test_{sysname}_{runname}', **kwargs)

                
script = 'sbatch.sh'
pyscript = 'pipeline.py'
sysdir = 'systems' 
sysnames = ['1btl',]
runs = ['mdrun_1', ]  # 'mdrun_2', 


# setup_cg_model(submit=False)
# md(submit=True, ntomp=8, mem='4G', q='public', p='htc', t='00-04:00:00',)
# extend(submit=True, ntomp=8, mem='2G', q='grp_sozkan', p='general', t='03-00:00:00',)
trjconv(submit=False)
# cluster(submit=False)
# rms_analysis(submit=True)
# dci_dfi(submit=True)
# tdlrt(submit=False)
# tdlrt_figs(submit=True)
# get_averages(submit=False)
# cov_analysis(submit=True)
# plot(submit=False)
# test(submit=True)

