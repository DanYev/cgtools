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

#%%
# Preparing the files
mdsys = GmxSystem('tests/test', 'test')
mdsys.prepare_files()
mdsys.sort_input_pdb("tests/1btl.pdb")

#%%
# # 2. Clean the PDB and split chains using GROMACS.
# mdsys.clean_pdb_gmx(in_pdb=mdsys.inpdb, clinput="8\n 7\n", ignh="no", renum="yes")
# mdsys.split_chains()
# mdsys.clean_chains_gmx(clinput="8\n 7\n", ignh="yes", renum="yes")
# mdsys.get_go_maps(append=True)

# # 3. Coarse-grain the proteins and RNA.
# mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)
# mdsys.martinize_rna(ef=200, el=0.3, eu=1.2, p="backbone", pf=500, append=True)
# mdsys.make_martini_topology_file(add_resolved_ions=False, prefix="chain")
# mdsys.make_cgpdb_file(bt="dodecahedron", d="1.2")

# # 4. Solvate and add ions.
# solvent = os.path.join(mdsys.wdir, "water.gro")
# mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
# mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")

# # 5. Create index files.
# mdsys.make_sys_ndx(backbone_atoms=["BB", "BB1", "BB3"])


