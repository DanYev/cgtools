#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup

Author: DY
"""

import os
import sys
import numpy as np
import pandas as pd
import shutil
import MDAnalysis as mda
from pathlib import Path
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *  # Assuming this imports required utilities

cli.run('rm -rf tests/test/*')
#%%
# First, we need to initialize an instance of GmxSystem which will take care of pathing
# and necessary files. This will use parent directory 'tests' relative to the current directory
# and root directory 'test' in 'tests' for our system
mdsys = GmxSystem(sysdir='tests', sysname='test')
#%%
# This command will actually prepare the necessary files
mdsys.prepare_files()
#%%
# This will sort chains and atoms in our PDB to avoid conflicts in the future and make file 'inpdb.pdb',
# which we can access later as mdsys.inpdb
mdsys.sort_input_pdb("../1btl.pdb")
print(mdsys.inpdb)

#%%
# Even though we don't need to have multiple chains in this case, this command splits our mdsys.inpdb
# into separate chains and moves Proteins and RNA/DNA to their respective directories
mdsys.split_chains()

#%%
# Coarse-grain the proteins using martinize2 by Martini
mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)
# Let's take a look at generated files:

# # 4. Solvate and add ions.
# solvent = os.path.join(mdsys.wdir, "water.gro")
# mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
# mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")

# # 5. Create index files.
# mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


