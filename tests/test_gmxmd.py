#!/usr/bin/env python
"""
Test Suite for reforge.gmxmd Package
=====================================

This module contains unit tests for the functions provided by the 
`reforge.gmxmd` package (and related CLI commands). These tests verify
the correct behavior of file preparation, PDB sorting, Gromacs command 
execution, PDB cleaning, chain splitting, and other functionality related 
to setting up molecular dynamics (MD) simulations with Gromacs.

Usage:
    Run the tests with pytest from the project root:

        pytest -v tests/test_gmxmd.py

Author: DY
"""

import os
import numpy as np
import pytest
from reforge.gmxmd import *
from reforge.cli import run

# Create a gmxSystem instance for testing.
mdsys = GmxSystem('tests', 'test')
in_pdb = '../dsRNA.pdb'

def test_prepare_files():
    """
    Test that mdsys.prepare_files() correctly prepares the file structure.
    """
    run('rm -rf tests/test/')
    mdsys.prepare_files()

def test_sort_input_pdb():
    """
    Test that sort_input_pdb() properly sorts and renames the input PDB file.
    """
    mdsys.sort_input_pdb(in_pdb)
    assert 'inpdb.pdb' in os.listdir(mdsys.root)

def test_gmx():
    """
    Test that mdsys.gmx() executes without error.
    """
    mdsys.gmx('')

def test_clean_pdb_gmx():
    """
    Test that clean_pdb_gmx() processes the PDB file as expected.
    """
    mdsys.clean_pdb_gmx(clinput='6\n7\n', ignh='yes')

def test_split_chains():
    """
    Test that split_chains() outputs chain files as expected.
    """
    mdsys.split_chains()
    assert 'chain_A.pdb' in os.listdir(mdsys.nucdir)
    assert 'chain_B.pdb' in os.listdir(mdsys.nucdir)

def test_clean_chains_gmx():
    """
    Test that clean_chains_gmx() processes chain PDB files as expected.
    """
    mdsys.clean_chains_gmx(clinput='6\n7\n', ignh='yes')
    assert 'chain_A.pdb' in os.listdir(mdsys.nucdir)
    assert 'chain_B.pdb' in os.listdir(mdsys.nucdir)

def test_martinize_rna():
    """
    Test that martinize_rna() executes without error.
    """
    mdsys.martinize_rna()

def test_make_solute_pdb():
    """
    Test that make_solute_pdb() creates the solute PDB file.
    """
    mdsys.make_solute_pdb()

def test_make_system_top():
    """
    Test that make_system_top() creates the system topology file.
    """
    mdsys.make_system_top()

def test_solvate():
    """
    Test that solvate() executes without error.
    """
    mdsys.solvate()

def test_add_bulk_ions():
    """
    Test that add_bulk_ions() adds ions to the system.
    """
    mdsys.add_bulk_ions()

def test_make_system_ndx():
    """
    Test that make_system_ndx() creates the system index file.
    """
    mdsys.make_system_ndx()

if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
    run('rm -rf tests/test/')