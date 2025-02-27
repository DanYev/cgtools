import os
import numpy as np
import pytest
from reforge.gmxmd import *


mdsys = gmxSystem('tests', 'test')
in_pdb = '1btl.pdb'


def test_sort_input_pdb(in_pdb):
    mdsys.sort_input_pdb(in_pdb)


def test_gmx():
    mdsys.gmx('')




if __name__ == '__main__':
    # pytest.main([os.path.abspath(__file__)])
    # test_sort_input_pdb(in_pdb)
    # test_gmx()
    

