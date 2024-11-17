import os
import subprocess as sp
import shutil
from contextlib import contextmanager
from functools import wraps

GMX = 'gmx_mpi'

##############################################################
# Some helper functions
##############################################################

@contextmanager
def change_directory(new_dir):
    """
    Context manager to temporarily change the working directory.

    Parameters:
    new_dir (str): The directory to switch to temporarily.
    """
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)
        
        
def from_wdir(func):
    """
    Decorator to temporarily change the working directory before executing a function.
    
    The first argument of the decorated function is expected to be the working directory (wdir).
    """
    @wraps(func)
    def wrapper(wdir, *args, **kwargs):
        with change_directory(wdir):  # Change to the given directory
            return func(wdir, *args, **kwargs)  # Execute the decorated function
    return wrapper
    
    
def args_to_str(*args):
    """
    Convert the positional arguments into a space-separated string.

    Parameters:
    *args: str
        Positional arguments to be converted.

    Returns:
    str: 
        A space-separated string of the arguments.
    """
    return ' '.join([str(arg) for arg in args])


def kwargs_to_str(hyphen='-', **kwargs):
    """
    Convert keyword arguments of the form 'key=value' into a string of '-key value'.

    Parameters:
    **kwargs: dict
        Keyword arguments to be converted.

    Returns:
    str:
        A string of keyword arguments formatted as '-key value'.
    """
    return ' '.join([f'{hyphen}{key} {value}' for key, value in kwargs.items()])    


def set_defaults(kwargs, defaults):
    """
    Set default values for kwargs if not already provided.
    
    Parameters:
    kwargs: dict
        The current keyword arguments passed to the function.
    defaults: dict
        The default values for the function.
        
    Returns:
    dict:
        The updated kwargs with defaults set.
    """
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return kwargs

##############################################################
# Generic functions
##############################################################

def run(*args, **kwargs):
    """
    Run a command line command from a Python script.

    Parameters:
    *args: str
        Positional arguments that form the command to be run.
    **kwargs: dict
        Keyword arguments that specify options for the command.
        Special keys:
        - 'clinput' (str, optional): Input to be passed to the command's stdin.
        - 'cltext' (bool, optional): Whether to treat the input as text (default is True).
    """
    clinput = kwargs.pop('clinput', None)
    cltext = kwargs.pop('cltext', True)
    command = args_to_str(*args) + ' ' + kwargs_to_str(**kwargs)
    sp.run(command.split(), input=clinput, text=cltext, check=False)
    # try:
    #     sp.run(command.split(), input=clinput, text=cltext, check=True)
    # except sp.CalledProcessError as e:
    #     raise RuntimeError(f"Command '{command}' failed with return code {e.returncode}") from e


@from_wdir
def run_gmx(wdir, command, **kwargs):
    """
    Run a GROMACS command from a Python script, switching to the specified working directory.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    command (str): 
        The GROMACS command to run (e.g., 'editconf', 'solvate').
    **kwargs: dict
        Additional options and flags for the GROMACS command.
        Special keys:
        - 'clinput' (str, optional): Input to be passed to the command's stdin.
        - 'cltext' (bool, optional): Whether to treat the input as text.
    """
    clinput = kwargs.pop('clinput', None)
    cltext = kwargs.pop('cltext', None)
    command = GMX + ' ' + command + ' ' + kwargs_to_str(**kwargs)
    sp.run(command.split(), input=clinput, text=cltext)
    
    
def sbatch(script, *args, **kwargs):
    """
    Submit a command as an sbatch script to SLURM scheduler.

    Parameters:
    script: str
        Shell script to run
    *args: str
        Positional arguments that form the command to be run inside the sbatch script.
    **kwargs: dict
        Keyword arguments that specify sbatch options.
        Special keys:
        - 'sbatch_opts' (dict, optional): Additional options for sbatch command like time, mem, etc.
        - 'clinput' (str, optional): Input to be passed to the command's stdin.
        - 'cltext' (bool, optional): Whether to treat the input as text (default is True).
        
    Example usage:
    sbatch('script.sh', var1, var2, t='01:00:00', mem='4G', N=1, c=4)
    """
    defaults = {
        'partition': 'htc',
        'qos': 'public',
        'N': 1,
        'n': 1,
        'c': 1,
        'mem': '1G',
        't': '01:00:00'
    }
    # Separate long and short options
    long_options = {key: value for key, value in kwargs.items() if len(key) > 1}
    short_options = {key: value for key, value in kwargs.items() if len(key) == 1}
    
    # Build the sbatch command string
    # sbatch_long_opts = kwargs_to_str(hyphen='--', **long_options)
    sbatch_long_opts = ' '.join([f'--{key.replace("_", "-")}={value}' for key, value in long_options.items()])
    sbatch_short_opts = kwargs_to_str(hyphen='-', **short_options)
    command = ' '.join(['sbatch' , sbatch_short_opts, sbatch_long_opts, str(script), args_to_str(*args)])
    sp.run(command.split())

    

##############################################################
# GROMACS functions
############################################################## 

@from_wdir
def editconf(wdir, **kwargs):
    """
    Run the GROMACS 'editconf' command to modify the configuration of the system.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    **kwargs: dict
        Additional options and flags for the 'editconf' command.
        Defaults:
        - 'f': 'system.pdb' (input file)
        - 'o': 'system.pdb' (output file)
        - 'bt': 'dodecahedron' (box type)
        - 'd': '1.25' (distance from the box edge)
    """
    defaults = {
        'f': 'system.pdb',
        'o': 'system.pdb',
        'bt': 'triclinic',
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'editconf', **kwargs)


@from_wdir
def solvate(wdir, **kwargs):
    """
    Run the GROMACS 'solvate' command to solvate the system with water.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    **kwargs: dict
        Additional options and flags for the 'solvate' command.
        Defaults:
        - 'cp': 'system.pdb' (input configuration file)
        - 'cs': 'water.gro' (input solvent structure file)
        - 'p': 'system.top' (input topology file)
        - 'o': 'system.pdb' (output configuration file)
        - 'radius': '0.23' (minimal distance between solute and solvent)
    """
    defaults = {
        'cp': 'system.pdb',
        'cs': 'water.gro',
        'p': 'system.top',
        'o': 'system.pdb',
        'radius': '0.23'
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'solvate', **kwargs)


@from_wdir
def make_ndx(wdir, clinput=None, **kwargs):
    """
    Run the GROMACS 'make_ndx' command to create an index file.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string passed to the GROMACS command for defining groups.
    **kwargs: dict
        Additional options and flags for the 'make_ndx' command.
        Defaults:
        - 'f': 'system.pdb' (input configuration file)
        - 'o': 'index.ndx' (output index file)
    """
    defaults = {
        'f': 'system.pdb',
        'o': 'index.ndx',
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'make_ndx', clinput=clinput, cltext=True, **kwargs)


@from_wdir
def grompp(wdir, **kwargs):
    """
    Run the GROMACS 'grompp' command to preprocess the input files and generate the `.tpr` file.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    **kwargs: dict
        Additional options and flags for the 'grompp' command.
        Defaults:
        - 'f': '../mdp/em.mdp' (input parameter file)
        - 'c': '../system.pdb' (input configuration file)
        - 'r': '../system.pdb' (input structure for position restraints)
        - 'p': '../system.top' (input topology file)
        - 'o': 'em.tpr' (output run input file)
        - 'maxwarn': '1' (maximum number of warnings allowed)
    """
    defaults = {
        'f': '../mdp/em.mdp',
        'c': '../system.pdb',
        'r': '../system.pdb',
        'p': '../system.top',
        'o': 'em.tpr',
        'maxwarn': '1'
    }
    kwargs = set_defaults(kwargs, defaults)
    command = f"gmx_mpi grompp -f {kwargs['f']} -c {kwargs['c']} -r {kwargs['r']} -p {kwargs['p']} -o {kwargs['o']} -maxwarn {kwargs['maxwarn']}"
    sp.run(command.split(), check=True)    
    
    
@from_wdir
def mdrun(wdir, **kwargs):
    """
    Run the GROMACS 'mdrun' command to perform molecular dynamics or energy minimization.

    Parameters:
    wdir (str): 
        The working directory where the energy minimization will be performed.
    **kwargs: dict
        Additional options and flags for the 'mdrun' command.
        Defaults:
        - 'deffnm': 'em' (output file base name)
        - 'ncpus': '0' (number of CPU threads to use; default is 0 which allows GROMACS to decide)
        - 'pin': 'on' (CPU pinning)
        - 'pinstride': '1' (pinning stride)
        - 'nsteps': '-2' (run for the length defined in the .mdp file)
    """
    defaults = {
        'deffnm': 'em',
        'ntomp': '6',
        'pin': 'on',
        'pinstride': '1',
        'nsteps': '-2'
    }
    kwargs = set_defaults(kwargs, defaults)
    options = f"-ntomp {kwargs['ntomp']} -pin {kwargs['pin']} -pinstride {kwargs['pinstride']} -nsteps {kwargs['nsteps']}"
    command = f"gmx_mpi mdrun {options} -deffnm {kwargs['deffnm']}"
    sp.run(command.split(), check=True)


@from_wdir
def trjconv(wdir, clinput='1\n1\n', **kwargs):
    """
    Run the GROMACS 'trjconv' command to convert trajectory files.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string specifying selections (default '1\n1\n').
    **kwargs: dict
        Additional options and flags for the 'trjconv' command.
        Defaults:
        - 's': 'md.tpr' (input run file)
        - 'f': 'md.xtc' (input trajectory file)
        - 'o': 'mdc.xtc' (output trajectory file)
        - 'fit': 'none' (fit method)
        - 'pbc': 'none' (periodic boundary conditions treatment)
        - 'ur': 'rect' (unit cell representation)
        - 'b': '0' (beginning time)
        - 'dt': '0' (time step)
    """
    defaults = {
        's': 'md.tpr',
        'f': 'md.xtc',
        'o': 'mdc.xtc',
        'fit': 'none',
        'pbc': 'none',
        'ur': 'rect',
        'b': '0',
        'dt': '0'
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'trjconv', clinput=clinput, cltext=True, **kwargs)


@from_wdir
def rmsf(wdir, clinput=None, **kwargs):
    """
    Run the GROMACS 'rmsf' command to calculate root mean square fluctuation (RMSF).

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string specifying atom selections.
    **kwargs: dict
        Additional options and flags for the 'rmsf' command.
        Defaults:
        - 's': 'md.tpr' (input run file)
        - 'f': 'mdc.xtc' (input trajectory file)
        - 'o': 'rms_analysis/rmsf.xvg' (output file)
        - 'b': '0' (beginning time)
        - 'xvg': 'none' (output format)
        - 'res': 'yes' (calculate per-residue RMSF)
    """
    defaults = {
        's': 'md.tpr',
        'f': 'mdc.xtc',
        'o': 'rms_analysis/rmsf.xvg',
        'b': '0',
        'xvg': 'none',
        'res': 'yes'
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'rmsf', clinput=clinput, cltext=True, **kwargs)


@from_wdir
def rms(wdir, clinput=None, **kwargs):
    """
    Run the GROMACS 'rms' command to calculate root mean square deviation (RMSD).

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string specifying atom selections.
    **kwargs: dict
        Additional options and flags for the 'rms' command.
        Defaults:
        - 's': 'md.tpr' (input run file)
        - 'f': 'mdc.xtc' (input trajectory file)
        - 'o': 'rms_analysis/rmsd.xvg' (output file)
        - 'b': '0' (beginning time)
        - 'xvg': 'none' (output format)
    """
    defaults = {
        's': 'md.tpr',
        'f': 'mdc.xtc',
        'o': 'rms_analysis/rmsd.xvg',
        'b': '0',
        'xvg': 'none'
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'rms', clinput=clinput, cltext=True, **kwargs)


@from_wdir
def rdf(wdir, clinput=None, **kwargs):
    """
    Run the GROMACS 'rdf' command to calculate  calculates radial distribution functions (RDF).

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string specifying atom selections.
    **kwargs: dict
        Additional options and flags for the 'rms' command.
        Defaults:
        - 's': 'md.tpr' (input run file)
        - 'f': 'mdc.xtc' (input trajectory file)
        - 'o': 'rms_analysis/rdf.xvg' (output file)
        - 'b': '0' (beginning time)
        - 'xvg': 'none' (output format)
    """
    defaults = {
        's': 'md.tpr',
        'f': 'mdc.xtc',
        'o': 'rms_analysis/rdf.xvg',
        'b': '0',
        'xvg': 'none'
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'rdf', clinput=clinput, cltext=True, **kwargs)


@from_wdir
def covar(wdir, clinput=None, **kwargs):
    """
    Run the GROMACS 'covar' command to get covariance matrix and normal modes.

    Parameters:
    wdir (str): 
        The working directory where the command should be executed.
    clinput (str, optional): 
        Input string specifying atom selections.
    **kwargs: dict
        Additional options and flags for the 'covar' command.
        Defaults:
        - 's': 'md.tpr' (input run file)
        - 'f': 'mdc.xtc' (input trajectory file)
        - 'o': 'rms_analysis/rmsd.xvg' (output file)
        - 'b': '0' (beginning time)
        - 'xvg': 'none' (output format)
    """
    defaults = {
        's': 'md.tpr',
        'f': 'mdc.xtc',
        'ascii': 'cov_analysis/covar.dat',
        'b': '0',
        'last': '10',
    }
    kwargs = set_defaults(kwargs, defaults)
    run_gmx(wdir, 'covar', clinput=clinput, cltext=True, **kwargs)
    
##############################################################
# JUNK
##############################################################   

    
def get_covariance_matrix(wdir, tb=100, te=500, tw=5, td=5):
    bdir = os.getcwd()
    os.chdir(wdir)
    os.chdir('analysis')
    for t in range(tb, te-tw-td, td):
        b = t
        e = t + tw
        command = f'gmx_mpi trjconv -s ../mdrun/md.tpr -f ../mdrun/md.xtc -o mdc_{b}.xtc -fit rot+trans -b {b} -e {e} -tu ns'
        sp.run(command.split(), input='1\n1\n', text=True)
        command = f'gmx_mpi covar -s md.tpr -f mdc_{b}.xtc -ascii covar_{b}.dat -b {b} -e {e} -tu ns -last 50'
        sp.run(command.split(), input='1\n3\n', text=True)
    files_to_delete = [f for f in os.listdir() if f.startswith('#')]
    for f in files_to_delete:
        os.remove(f)
    os.remove(f'covar_{te}.dat')
    os.chdir(bdir)
    
    
    