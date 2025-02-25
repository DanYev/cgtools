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
DEBUG = os.environ.get('DEBUG', '0') == '1'
log_level = logging.DEBUG if DEBUG else logging.INFO
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=log_level,
    format='[%(levelname)s] %(message)s'
)
# logger.debug("Debug mode is enabled.")
# logger.info("Logger is set up.")


def timeit(func):
    """
    A decorator to measure the execution time of a function.
    Parameters:
        func (callable): The function to be timed.
    Returns:
        callable: A wrapped function that prints its execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()  # End the timer
        execution_time = end_time - start_time
        logger.debug(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper


def memprofit(func):
    """
    A decorator to memory profile a function.
    Parameters:
        func (callable): The function to be timed.
    Returns:
        callable: A wrapped function that prints its execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()             # Start the profiles
        result = func(*args, **kwargs)  # Call the original function
        current, peak = tracemalloc.get_traced_memory()  # Get the current and peak memory usage
        logger.debug(f"Memory usage after executing '{func.__name__}': {current/1024**2:.2f} MB, Peak memory usage: {peak/1024**2:.2f} MB")
        tracemalloc.stop()
        return result
    return wrapper 


@contextmanager
def cd(newdir):
    """Context manager for changing the current working directory."""
    prevdir = Path.cwd()
    os.chdir(newdir)
    logger.info(f'Working directory: {newdir}')
    try:
        yield
    finally:
        clean_dir()
        os.chdir(prevdir)


def clean_dir(directory=".", pattern="#*"):
    """Clean some of the annoying GROMACS backups."""
    directory = Path(directory)
    for file_path in directory.glob(pattern):    
        if file_path.is_file():
            file_path.unlink()     


def cuda_info():
    if cp.cuda.is_available():
        logger.info("CUDA is available")
        device_count = cp.cuda.runtime.getDeviceCount()
        logger.info("Number of CUDA devices:", device_count)
        return True
    else:
        logger.info("CUDA is not available")
        return False


def cuda_detected():
    if cp.cuda.is_available():
        return True
    else:
        logger.info("CUDA is not available")
        return False


def percentile(x):
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px            


def count_itp_atoms(file_path):
    in_atoms_section = False
    atom_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip whitespace and check if it's a comment or empty line
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                # Detect the start of the [ atoms ] section
                if line.startswith("[ atoms ]"):
                    in_atoms_section = True
                    continue
                # Detect the start of a new section
                if in_atoms_section and line.startswith('['):
                    break
                # Count valid lines in the [ atoms ] section
                if in_atoms_section:
                    atom_count += 1
        return atom_count
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0