#!/usr/bin/env python
"""
Simple CG Protein
=================

Test Go-Martini setup

Requirements:
    - GROMACS
    - Python 3.x

Author: DY
"""
from pathlib import Path
import shutil
import pytest
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

# Create a gmxSystem instance for testing.
mdsys = GmxSystem("tests", "test")
mdrun = GmxRun("tests", "test", "test")
test_dir = Path("tests") / "test"
test_pdb = Path(test_dir / ".." / "1btl.pdb").resolve()
if test_dir.exists():
    shutil.rmtree(test_dir)

def test_prepare():
    mdsys.prepare_files()
    mdsys.sort_input_pdb(test_pdb)
    mdsys.split_chains()

def test_martini_en():
    mdsys.martinize_proteins_en(ef=700, el=0.0, eu=0.9, p='backbone', pf=500, append=False)

def test_martini_go():
    mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)

if __name__ == '__main__':
    pytest.main([str(Path(__file__).resolve())])
