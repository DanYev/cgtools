import os
import numpy as np
import pytest
from reforge.gmxmd import *
from reforge.cli import run


mdsys = gmxSystem('tests', 'test')
in_pdb = '../dsRNA.pdb'


def test_prepare_files():
    run('rm -rf tests/test/')
    mdsys.prepare_files()


def test_sort_input_pdb():
    mdsys.sort_input_pdb(in_pdb)
    assert 'inpdb.pdb' in os.listdir(mdsys.root)


def test_gmx():
    mdsys.gmx('')


def test_clean_pdb_gmx():
    mdsys.clean_pdb_gmx(clinput='6\n7\n', ignh='yes') # 


def test_split_chains():
    mdsys.split_chains()
    assert 'chain_A.pdb' in os.listdir(mdsys.nucdir)
    assert 'chain_B.pdb' in os.listdir(mdsys.nucdir)


def test_clean_chains_gmx():
    mdsys.clean_chains_gmx(clinput='6\n7\n', ignh='yes')
    assert 'chain_A.pdb' in os.listdir(mdsys.nucdir)
    assert 'chain_B.pdb' in os.listdir(mdsys.nucdir)


def test_martinize_rna():
    mdsys.martinize_rna()


def test_make_solute_pdb():
    mdsys.make_solute_pdb()


def test_make_system_top():
    mdsys.make_system_top()


def test_solvate():
    mdsys.solvate() 


def test_add_bulk_ions():
    mdsys.add_bulk_ions()


def test_make_system_ndx():
    mdsys.make_system_ndx()


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
    # test_prepare_files()
    # test_sort_input_pdb()
    # test_gmx()
    # test_clean_pdb_gmx()
    # test_split_chains()
    # test_clean_chains_gmx()
    # test_martinize_rna()
    # test_make_solute_pdb()
    # test_make_system_top()
    # test_solvate()
    # test_add_bulk_ions()
    # test_make_system_ndx()
