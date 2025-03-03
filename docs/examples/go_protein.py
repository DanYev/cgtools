#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup

Author: DY
"""
import os
import sys
import shutil
import MDAnalysis as mda
from pathlib import Path
from reforge import cli, io, mdm
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import *  # Assuming this imports required utilities


WDIR = '.' # '.' for html, 'examples' for manual
os.chdir(WDIR)
cli.run('rm -rf test/*')
#%%
# First, we need to initialize an instance of GmxSystem which will take care of pathing
# and necessary files. This will use parent directory 'tests' relative to the current directory
# and root directory 'test' in 'tests' for our system
mdsys = GmxSystem(sysdir='.', sysname='test')
#%%
# This command will actually prepare the necessary files
mdsys.prepare_files()
for f in mdsys.root.iterdir():
    print(f)
#%%
# This will sort chains and atoms in our PDB to avoid conflicts in the future and make file 'inpdb.pdb',
# which we can access later as mdsys.inpdb
in_pdb = "../1btl.pdb" # relatrive to mdsys.root
mdsys.sort_input_pdb(in_pdb)
print(mdsys.inpdb)

#%%
# Even though we don't need to have multiple chains in this case, this command splits our mdsys.inpdb
# into separate chains and moves Proteins and RNA/DNA to their respective directories
mdsys.split_chains()

#%%
# Coarse-grain the proteins using martinize2 by Martini
mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)

#%%
# Let's take a look at generated files. The topology should contain the main protein topology file 'chain_A.itp'
# and virtual sites parameters for go-model - 'go_atomtypes.itp' and 'go_nbparams.itp'
for f in mdsys.topdir.iterdir():
    print(f)

#%% 
# And the coarse-grained structure shoulbe in 'cgdir'
for f in mdsys.cgdir.iterdir():
    print(f)

#%% 
# These two commands will combine all topology and structre files. 
mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
# make_cg_structure uses 'gmx pdb2gmx' module to make the simulation box.
# the description of the module can be found online
mdsys.make_cg_structure(bt='dodecahedron', d='1.2', ) # CG structure. Returns mdsys.solupdb ("solute.pdb") file    

#%% 
# Now we need to add solvent and neutralize the system's charge
solvent = mdsys.root / "water.gro"
mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")

#%% 
# In order to work with GROMACS' selections we need to make a .ndx file mdsys.sysndx
# Order of the groups: 1.System 2.Solute 3.Backbone 4.Solvent 5. Not Water 6...chains... 
# Can add custom groups using AtomList.write_to_ndx() method
mdsys.make_system_ndx(backbone_atoms=["BB"])


