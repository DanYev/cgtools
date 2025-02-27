import os
import numpy as np
import pytest
from reforge.gmxmd import *


mdsys = gmxSystem('tests', 'test')
in_pdb = '1btl.pdb'


def test_sort_input_pdb():
    mdsys.sort_input_pdb(in_pdb)
    assert 'inpdb.pdb' in os.listdir(mdsys.root)

def test_gmx():
    mdsys.gmx('')


def test_clean_pdb_gmx():
    mdsys.clean_pdb_gmx(clinput='6\n7\n', ignh='yes') # 


def test_split_chains():
    in_pdb = 'dsRNA.pdb'
    mdsys.sort_input_pdb(in_pdb)
    os.makedirs(mdsys.nucdir, exist_ok=True)
    mdsys.split_chains()
    assert 'chain_A.pdb' in os.listdir(mdsys.nucdir)
    assert 'chain_B.pdb' in os.listdir(mdsys.nucdir)




if __name__ == '__main__':
    # pytest.main([os.path.abspath(__file__)])
    # test_sort_input_pdb(in_pdb)
    # test_gmx()
    test_clean_pdb_gmx()
    test_split_chains()


