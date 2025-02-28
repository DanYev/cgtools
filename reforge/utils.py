"""
File: utils.py
Description:
    This module provides utility functions and decorators for the reForge workflow.
    It includes decorators for timing and memory profiling functions, a context manager
    for changing the working directory, and helper functions for cleaning directories and
    detecting CUDA availability.

Usage Example:
    >>> from utils import timeit, memprofit, cd, clean_dir, cuda_info
    >>>
    >>> @timeit
    ... def my_function():
    ...     # Function implementation here
    ...     pass
    >>>
    >>> with cd("/tmp"):
    ...     # Perform operations in /tmp
    ...     pass
    >>>
    >>> cuda_info()

Requirements:
    - Python 3.x
    - cupy
    - Standard libraries: logging, os, sys, time, tracemalloc, contextlib, functools, pathlib

Author: Your Name
Date: YYYY-MM-DD
"""

import logging
import os
import sys
import time
import tracemalloc
import cupy as cp
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

# Use an environment variable (DEBUG=1) to toggle debug logging
DEBUG = os.environ.get("DEBUG", "0") == "1"
log_level = logging.DEBUG if DEBUG else logging.INFO
logger = logging.getLogger(__name__)
logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
# logger.debug("Debug mode is enabled.")
# logger.info("Logger is set up.")


def timeit(func):
    """Decorator to measure and log the execution time of a function.

    Parameters
    ----------
    func : callable
        The function whose execution time is to be measured.

    Returns
    -------
    callable
        A wrapped version of the input function that logs its execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start timer
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        logger.debug(
            f"Function '{func.__module__}.{func.__name__}' executed in {execution_time:.6f} seconds."
        )
        return result

    return wrapper


def memprofit(func):
    """Decorator to profile and log the memory usage of a function.

    Parameters
    ----------
    func : callable
        The function whose memory usage is to be profiled.

    Returns
    -------
    callable
        A wrapped version of the input function that logs its memory usage.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()  # Start memory tracking
        result = func(*args, **kwargs)  # Execute the function
        current, peak = (
            tracemalloc.get_traced_memory()
        )  # Get current and peak memory usage
        logger.debug(
            f"Memory usage after executing '{func.__module__}.{func.__name__}': {current/1024**2:.2f} MB, Peak: {peak/1024**2:.2f} MB"
        )
        tracemalloc.stop()  # Stop memory tracking
        return result

    return wrapper


@contextmanager
def cd(newdir):
    """Context manager to temporarily change the current working directory.

    Parameters
    ----------
    newdir : str or Path
        The target directory to change into.

    Yields
    ------
    None
        After the context is exited, the working directory is reverted to its original value.
    """
    prevdir = Path.cwd()
    os.chdir(newdir)
    logger.info(f"Changed working directory to: {newdir}")
    try:
        yield
    finally:
        os.chdir(prevdir)


def clean_dir(directory=".", pattern="#*"):
    """Remove files matching a specific pattern from a directory.

    Parameters
    ----------
    directory : str or Path, optional
        The directory in which to search for files (default is the current directory).
    pattern : str, optional
        The glob pattern for files to remove (default is "#*").

    Returns
    -------
    None
    """
    directory = Path(directory)
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def cuda_info():
    """Check CUDA availability and log CUDA device information if available.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    if cp.cuda.is_available():
        logger.info("CUDA is available")
        device_count = cp.cuda.runtime.getDeviceCount()
        logger.info("Number of CUDA devices: " + str(device_count))
        return True
    else:
        logger.info("CUDA is not available")
        return False


def cuda_detected():
    """Check if CUDA is detected without logging detailed device information.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    if cp.cuda.is_available():
        return True
    else:
        logger.info("CUDA is not available")
        return False
