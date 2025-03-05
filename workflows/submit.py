from reforge.cli import sbatch, run, dojob


def setup(submit=False, **kwargs): 
    """
    Set up the system by processing each system name.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments for the job.
    """
    kwargs.setdefault('mem', '3G')
    for sysname in sysnames:
        dojob(submit, script, pyscript, 'setup', sysdir, sysname, 
              J=f'setup_{sysname}', **kwargs)


def md(submit=True, ntomp=8, **kwargs):
    """
    Run molecular dynamics simulations for each system and run.
    
    Parameters:
        submit (bool): Whether to submit the job.
        ntomp (int): Number of OpenMP threads.
        **kwargs: Additional keyword arguments.
    """
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
    """Extend simulations by processing each system and run."""
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
    """Convert trajectories for each system and run."""
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'trjconv', sysdir, sysname, runname,
                  J=f'trjconv_{sysname}_{runname}', **kwargs)

            
def rms_analysis(submit=True, **kwargs):
    """Perform RMSD analysis for each system and run."""
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'rms_analysis', sysdir, sysname, runname,
                  J=f'rms_{sysname}_{runname}', **kwargs)


def cluster(submit=True, **kwargs):
    """Run clustering analysis for each system and run."""
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'cluster', sysdir, sysname, runname,
                  J=f'cluster_{sysname}_{runname}', **kwargs)    


def cov_analysis(submit=True, **kwargs):
    """Perform covariance analysis for each system and run."""
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'cov_analysis', sysdir, sysname, runname,
                  J=f'cov_{sysname}_{runname}', **kwargs)

                
def tdlrt_analysis(submit=True, **kwargs):
    """Perform tdlrt analysis for each system and run."""
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '30G')
    kwargs.setdefault('G', '1')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, 'tdlrt_analysis', sysdir, sysname, runname,
                  J=f'tdlrt_{sysname}_{runname}', **kwargs)
 

def get_averages(submit=False, **kwargs):
    """Calculate average values for each system."""
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        dojob(submit, script, pyscript, 'get_averages', sysdir, sysname, 
              J=f'av_{sysname}', **kwargs)


def get_td_averages(submit=False, **kwargs):
    """Calculate time-dependent averages for each system."""
    kwargs.setdefault('mem', '80G')
    for sysname in sysnames:
        dojob(submit, script, pyscript, 'get_td_averages', sysdir, sysname, 
              J=f'tdav_{sysname}', **kwargs)   


def plot(submit=False, **kwargs):
    """Generate plots for each system."""
    for sysname in sysnames:
        dojob(submit, script, 'plot.py', sysdir, sysname, 
              J='plotting', **kwargs)


def sysjob(jobname, submit=False, **kwargs):
    """
    Submit or run a system-level job for each system.
    
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        dojob(submit, script, pyscript, jobname, sysdir, sysname, 
              J=f'{sysname}_{jobname}', **kwargs)


def runjob(jobname, submit=False, **kwargs):
    """
    Submit or run a run-level job for each system and run.
    
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, script, pyscript, jobname, sysdir, sysname, runname,
                  J=f'{runname}_{jobname}', **kwargs)

             
script = 'sbatch.sh'
pyscript = 'egfr_pipe.py'
sysdir = 'systems' 
sysnames = ['egfr'] # 1btl 8aw3
runs = ['mdrun_1', 'mdrun_2', ] 


# setup(submit=False, mem='4G')
# md(submit=True, ntomp=8, mem='4G', q='grp_sozkan', p='general', t='04-00:00:00',)
# extend(submit=True, ntomp=8, mem='2G', q='grp_sozkan', p='general', t='03-00:00:00',)
# trjconv(submit=True)
# rms_analysis(submit=True)
# cov_analysis(submit=True)
get_averages(submit=True, mem='32G')
# plot(submit=False)
# cluster(submit=False)
# get_td_averages(submit=False)
# tdlrt_figs(submit=True)
# test(submit=True)
# sysjob('make_ndx', submit=False)
# runjob('runjob', submit=True, mem='16G')
