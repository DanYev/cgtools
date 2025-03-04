"""File: cli.py

Description:
    This module provides a collection of command-line interface utilities for executing
    shell commands, submitting SLURM sbatch jobs, and running GROMACS operations from within
    a Python script. It includes generic functions for running commands and managing directories,
    as well as specialized wrappers for GROMACS commands (e.g., editconf, solvate, grompp, mdrun,
    and others) for molecular dynamics analysis.

Usage Example:
    >>> from cli import run, sbatch, gmx, change_directory
    >>> # Run a simple shell command
    >>> run('ls', '-l')
    >>>
    >>> # Change directory temporarily
    >>> with change_directory('/tmp'):
    ...     run('pwd')
    >>>
    >>> # Submit a job via SLURM
    >>> sbatch('script.sh', 'arg1', 'arg2', t='01:00:00', mem='4G', N='1', c='4')
    >>>
    >>> # Execute a GROMACS command
    >>> gmx('editconf', f='system.pdb', o='system_out.pdb')

Requirements:
    - Python 3.x
    - Standard libraries: os, subprocess, shutil, contextlib, functools
    - SLURM (for sbatch)
    - GROMACS (for GROMACS wrappers)

Author: DY
Date: YYYY-MM-DD
"""

import os
import subprocess as sp
from contextlib import contextmanager
from functools import wraps


##############################################################
# Generic Functions
##############################################################


def run(*args, **kwargs):
    """Execute a shell command from within a Python script.

    Parameters
    ----------
    *args : str
        Positional arguments that compose the command to be executed.
    **kwargs : dict
        Additional keyword arguments for command options. Special keys:
          - clinput (str, optional): Input string to be passed to the command's standard input.
          - cltext (bool, optional): Whether the input should be treated as text (default True).

    Returns
    -------
    None
    """
    clinput = kwargs.pop("clinput", None)
    cltext = kwargs.pop("cltext", True)
    command = args_to_str(*args) + " " + kwargs_to_str(**kwargs)
    sp.run(command.split(), input=clinput, text=cltext, check=False)


def sbatch(script, *args, **kwargs):
    """Submit a shell script as a SLURM sbatch job.

    Parameters
    ----------
    script : str
        The path to the shell script to be executed.
    *args : str
        Additional positional arguments that are passed to the script.
    **kwargs : dict
        Additional keyword options for the sbatch command. Special keys include:
          - clinput (str, optional): Input string for the command's standard input.
          - cltext (bool, optional): Indicates if input should be treated as text (default True).

    Example
    -------
    >>> sbatch('script.sh', 'arg1', 'arg2', t='01:00:00', mem='4G', N='1', c='4')

    Returns
    -------
    None
    """
    kwargs.setdefault("t", "01:00:00")
    kwargs.setdefault("q", "public")
    kwargs.setdefault("p", "htc")
    kwargs.setdefault("N", "1")
    kwargs.setdefault("n", "1")
    kwargs.setdefault("c", "1")
    kwargs.setdefault("mem", "2G")
    kwargs.setdefault("e", "slurm_jobs/error.%A.err")
    kwargs.setdefault("o", "slurm_jobs/output.%A.out")
    # Separate long and short options
    long_options = {
        key: value for key,
        value in kwargs.items() if len(key) > 1}
    short_options = {
        key: value for key,
        value in kwargs.items() if len(key) == 1}
    # Build the sbatch command string
    sbatch_long_opts = " ".join(
        [f'--{key.replace("_", "-")}={value}' for key,
         value in long_options.items()]
    )
    sbatch_short_opts = kwargs_to_str(hyphen="-", **short_options)
    command = " ".join(
        ["sbatch",
         sbatch_short_opts,
         sbatch_long_opts,
         str(script),
            args_to_str(*args)]
    )
    sp.run(command.split(), check=True)


def dojob(submit, *args, **kwargs):
    """Submit or run a job based on the 'submit' flag.

    This function provides a simple interface to either submit a job to SLURM
    (using the 'sbatch' command) or to run it locally via bash. When `submit` is
    True, the function calls the `sbatch` function with the given arguments and
    keyword options, which handles setting SLURM parameters and submitting the job.
    When `submit` is False, the job is executed immediately using bash.

    Parameters
    ----------
    submit : bool
        If True, submit the job to SLURM using sbatch; if False, run the job locally via bash.
    *args : tuple of str
        Positional arguments representing the script and any additional command-line 
        arguments that should be passed to the job.
    **kwargs : dict
        Keyword arguments for job configuration. These are passed to the `sbatch` function
        when submitting the job. They can include SLURM options (such as 't' for time, 'mem' for memory,
        etc.) as well as any special keys recognized by `sbatch` (e.g., 'clinput' for standard input).
    
    Examples
    --------
    To submit a job to SLURM:
    
    >>> dojob(True, 'script.sh', 'arg1', 'arg2', t='01:00:00', mem='4G', N='1', c='4')
    
    To run the job locally via bash:
    
    >>> dojob(False, 'script.sh', 'arg1', 'arg2')
    
    Returns
    -------
    None
    """
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)


def gmx(command, gmx_callable="gmx_mpi", **kwargs):
    """Execute a GROMACS command.

    Parameters
    ----------
    command : str
        The GROMACS command to execute (e.g., 'editconf', 'solvate').
    gmx_callable : str, optional
        The GROMACS executable to use (default is 'gmx_mpi').
    **kwargs : dict
        Additional options for the command. Special keys:
          - clinput (str, optional): Input to be passed to the command's standard input.
          - cltext (bool, optional): Whether to treat the input as text (default True).

    Returns
    -------
    None
    """
    clinput = kwargs.pop("clinput", None)
    cltext = kwargs.pop("cltext", True)
    command = gmx_callable + " " + command + " " + kwargs_to_str(**kwargs)
    sp.run(command.split(), input=clinput, text=cltext, check=True)


##############################################################
# Utility Functions
##############################################################


@contextmanager
def change_directory(new_dir):
    """Temporarily change the working directory.

    Parameters
    ----------
    new_dir : str
        The directory path to change into.

    Yields
    ------
    None
        After executing the enclosed block, reverts to the original directory.
    """
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


def from_wdir(func):
    """Decorator to temporarily change the working directory before executing a
    function.

    The first argument of the decorated function should be the target working directory.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function that executes in the specified directory.
    """

    @wraps(func)
    def wrapper(wdir, *args, **kwargs):
        with change_directory(wdir):
            return func(wdir, *args, **kwargs)

    return wrapper


##############################################################
# GROMACS Functions
##############################################################


def gmx_editconf(**kwargs):
    """Execute the GROMACS 'editconf' command to modify system configuration.

    Parameters
    ----------
    wdir : str
        The working directory where the command is executed.
    **kwargs : dict
        Additional options for 'editconf'. Defaults are:
          - f: 'solute.pdb' (input file)
          - o: 'solute.pdb' (output file)
          - bt: 'triclinic' (box type)

    Returns
    -------
    None
    """
    kwargs.setdefault("f", "solute.pdb")
    kwargs.setdefault("o", "solute.pdb")
    kwargs.setdefault("bt", "triclinic")
    gmx("editconf", **kwargs)


def gmx_solvate(**kwargs):
    """Execute the GROMACS 'solvate' command to add solvent to the system.

    Parameters
    ----------
    wdir : str
        The working directory where the command is executed.
    **kwargs : dict
        Additional options for 'solvate'. Defaults are:
          - cp: 'solute.pdb' (input configuration file)
          - cs: 'water.gro' (solvent structure file)
          - p: 'system.top' (topology file)
          - o: 'system.pdb' (output configuration file)
          - radius: '0.23' (minimal solute-solvent distance)

    Returns
    -------
    None
    """
    kwargs.setdefault("cp", "solute.pdb")
    kwargs.setdefault("cs", "water.gro")
    kwargs.setdefault("p", "system.top")
    kwargs.setdefault("o", "system.pdb")
    kwargs.setdefault("radius", "0.23")
    gmx("solvate", **kwargs)


def gmx_make_ndx(clinput=None, **kwargs):
    """Execute the GROMACS 'make_ndx' command to create an index file.

    Parameters
    ----------
    clinput : str, optional
        Input string for defining index groups.
    **kwargs : dict
        Additional options for 'make_ndx'. Defaults are:
          - f: 'system.pdb' (input configuration file)
          - o: 'index.ndx' (output index file)

    Returns
    -------
    None
    """
    kwargs.setdefault("f", "system.pdb")
    kwargs.setdefault("o", "index.ndx")
    gmx("make_ndx", clinput=clinput, cltext=True, **kwargs)


def gmx_grompp(**kwargs):
    """Execute the GROMACS 'grompp' command to preprocess input files and
    generate a .tpr file.

    Parameters
    ----------
    **kwargs : dict
        Additional options for 'grompp'. Defaults are:
          - maxwarn: '1'
          - f: '../mdp/em.mdp' (input parameter file)
          - c: '../system.pdb' (input configuration file)
          - r: '../system.pdb' (input structure for position restraints)
          - p: '../system.top' (input topology file)
          - o: 'em.tpr' (output run file)

    Returns
    -------
    None
    """
    kwargs.setdefault("maxwarn", "1")
    gmx("grompp", **kwargs)


def gmx_mdrun(**kwargs):
    """Execute the GROMACS 'mdrun' command to perform molecular dynamics
    simulations or energy minimization.

    Parameters
    ----------
    **kwargs : dict
        Additional options for 'mdrun'. Defaults are:
          - ntomp: '8'
          - pin: 'on'
          - pinstride: '1'

    Returns
    -------
    None
    """
    kwargs.setdefault("ntomp", "8")
    kwargs.setdefault("pin", "on")
    kwargs.setdefault("pinstride", "1")
    gmx("mdrun", **kwargs)


def gmx_trjconv(clinput="1\n1\n", **kwargs):
    r"""Execute the GROMACS 'trjconv' command to convert trajectory files.

    Parameters
    ----------
    clinput : str, optional
        Input string for trajectory frame selection (default '1\n1\n').
    **kwargs : dict
        Additional options for 'trjconv'.

    Returns
    -------
    None
    """
    gmx("trjconv", clinput=clinput, cltext=True, **kwargs)


def gmx_rmsf(clinput=None, **kwargs):
    """Execute the GROMACS 'rmsf' command to compute root mean square
    fluctuations (RMSF).

    Parameters
    ----------
    wdir : str
        The working directory where the command is executed.
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'rmsf'. Defaults are:
          - s: 'md.tpr' (input run file)
          - f: 'mdc.xtc' (trajectory file)
          - o: 'rms_analysis/rmsf.xvg' (output file)
          - b: '0' (start time)
          - xvg: 'none' (output format)
          - res: 'yes' (calculate per-residue RMSF)

    Returns
    -------
    None
    """
    kwargs.setdefault("s", "md.tpr")
    kwargs.setdefault("f", "mdc.xtc")
    kwargs.setdefault("o", "rms_analysis/rmsf.xvg")
    kwargs.setdefault("b", "0")
    kwargs.setdefault("xvg", "none")
    kwargs.setdefault("res", "yes")
    gmx("rmsf", clinput=clinput, cltext=True, **kwargs)


def gmx_rms(clinput=None, **kwargs):
    """Execute the GROMACS 'rms' command to compute root mean square deviation
    (RMSD).

    Parameters
    ----------
    wdir : str
        The working directory where the command is executed.
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'rms'. Defaults are:
          - s: 'md.tpr' (input run file)
          - f: 'mdc.xtc' (trajectory file)
          - o: 'rms_analysis/rmsd.xvg' (output file)
          - b: '0' (start time)
          - xvg: 'none' (output format)
          - fit: 'rot+trans' (fitting method)

    Returns
    -------
    None
    """
    kwargs.setdefault("s", "md.tpr")
    kwargs.setdefault("f", "mdc.xtc")
    kwargs.setdefault("o", "rms_analysis/rmsd.xvg")
    kwargs.setdefault("b", "0")
    kwargs.setdefault("xvg", "none")
    kwargs.setdefault("fit", "rot+trans")
    gmx("rms", clinput=clinput, cltext=True, **kwargs)


def gmx_rdf(clinput=None, **kwargs):
    """Execute the GROMACS 'rdf' command to compute radial distribution
    functions (RDF).

    Parameters
    ----------
    wdir : str
        The working directory where the command is executed.
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'rdf'. Defaults are:
          - s: 'md.tpr' (input run file)
          - f: 'mdc.xtc' (trajectory file)
          - o: 'rms_analysis/rdf.xvg' (output file)
          - b: '0' (start time)
          - xvg: 'none' (output format)

    Returns
    -------
    None
    """
    kwargs.setdefault("s", "md.tpr")
    kwargs.setdefault("f", "mdc.xtc")
    kwargs.setdefault("o", "rms_analysis/rdf.xvg")
    kwargs.setdefault("b", "0")
    kwargs.setdefault("xvg", "none")
    gmx("rdf", clinput=clinput, cltext=True, **kwargs)


def gmx_cluster(clinput=None, **kwargs):
    """Execute the GROMACS 'cluster' command to perform clustering analysis.

    Parameters
    ----------
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'cluster'.

    Returns
    -------
    None
    """
    gmx("cluster", clinput=clinput, cltext=True, **kwargs)


def gmx_extract_cluster(clinput=None, **kwargs):
    """Execute the GROMACS 'extract-cluster' command to extract clustered
    structures.

    Parameters
    ----------
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'extract-cluster'.

    Returns
    -------
    None
    """
    gmx("extract-cluster", clinput=clinput, cltext=True, **kwargs)


def gmx_covar(clinput=None, **kwargs):
    """Execute the GROMACS 'covar' command to compute the covariance matrix and
    normal modes.

    Parameters
    ----------
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'covar'.

    Returns
    -------
    None
    """
    gmx("covar", clinput=clinput, cltext=True, **kwargs)


def gmx_anaeig(clinput=None, **kwargs):
    """Execute the GROMACS 'anaeig' command to analyze eigenvalues and
    eigenvectors.

    Parameters
    ----------
    clinput : str, optional
        Input string for defining groups.
    **kwargs : dict
        Additional options for 'anaeig'. Defaults include:
          - v: 'eigenvec.trr' (eigenvector file)

    Returns
    -------
    None
    """
    kwargs.setdefault("v", "eigenvec.trr")
    gmx("anaeig", clinput=clinput, cltext=True, **kwargs)


def gmx_make_edi(clinput=None, **kwargs):
    """Execute the GROMACS 'make_edi' command to generate essential dynamics
    (ED) information.

    Parameters
    ----------
    clinput : str, optional
        Input string for atom selection.
    **kwargs : dict
        Additional options for 'make_edi'. Defaults are:
          - f: 'eigenvec.trr' (input eigenvector file)

    Returns
    -------
    None
    """
    kwargs.setdefault("f", "eigenvec.trr")
    gmx("make_edi", clinput=clinput, cltext=True, **kwargs)


##############################################################
# Helper Functions
##############################################################


def args_to_str(*args):
    """Convert positional arguments to a space-separated string.

    Parameters
    ----------
    *args : str
        Positional arguments to be concatenated.

    Returns
    -------
    str
        A space-separated string representation of the arguments.
    """
    return " ".join([str(arg) for arg in args])


def kwargs_to_str(hyphen="-", **kwargs):
    """Convert keyword arguments to a formatted string with a given hyphen
    prefix.

    Parameters
    ----------
    hyphen : str, optional
        The prefix to use for each keyword (default is '-').
    **kwargs : dict
        Keyword arguments to be formatted.

    Returns
    -------
    str
        A formatted string of the keyword arguments.
    """
    return " ".join(
        [f"{hyphen}{key} {value}" for key, value in kwargs.items()])
