"""
Unit and regression test for the pyrotini package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pyrotini


def test_pyrotini_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyrotini" in sys.modules

def test_fix_go_map():
    fix_go_map('protein.map', out_map='go.map')
    
def test_make_topology_file():
    make_topology_file(test)
    
def test_link_itps():
    link_itps(test)

def test_martinize_go():
    martinize_go('protein.pdb', wdir=test, go_map='go.map',)
    
def test_solvate():
    solvate(test)

def test_energy_minimization():
    energy_minimization(test, ncpus)

def test_heatup():
    heatup(test, ncpus)
    
def test_equilibration():
    equilibration(test, ncpus)
    
def test_md():
    md(test, ncpus)
    
def test_convert_trajectory():
    convert_trajectory(test)
    
def test_prepare_files():
    prepare_files(test)
    
def test_get_covariance_matrix():
    get_covariance_matrix(test)
    