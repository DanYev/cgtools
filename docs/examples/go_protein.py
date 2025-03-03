#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup

Requirements:
    - GROMACS
    - Python 3.x

Author: DY
"""
import os
from reforge import cli
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

WDIR = '.' # '.' for html, 'examples' for manual
os.chdir(WDIR)
cli.run('rm -rf test/*')
#%%
# First, we need to initialize an instance of GmxSystem which will take care of pathing
# and necessary files. This will use parent directory 'tests' relative to the current directory
# and root directory 'test' in 'tests' for the system.
mdsys = GmxSystem(sysdir='.', sysname='test')
#%%
# This command will actually prepare the necessary files
mdsys.prepare_files()
for f in mdsys.root.iterdir():
    print(f)
#%%
# This will sort chains and atoms in our PDB to avoid conflicts in the future and make file 'inpdb.pdb',
# which we can access later as mdsys.inpdb
in_pdb = "../1btl.pdb" # relative to mdsys.root or absolute
mdsys.sort_input_pdb(in_pdb)
print(mdsys.inpdb)
#%%
# Even though we don't need to have multiple chains in this case, this command splits our mdsys.inpdb
# into separate chains and moves Proteins and RNA/DNA to their respective directories
mdsys.split_chains()
#%%
# Coarse-grain the proteins using martinize2 by Martini
mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False);
#%%
# Let's take a look at the generated files. The topology should contain the main protein topology file 'chain_A.itp'
# and virtual sites parameters for go-model - 'go_atomtypes.itp' and 'go_nbparams.itp'.
for f in mdsys.topdir.iterdir():
    print(f)
#%% 
# And the coarse-grained structure should be in 'cgdir'.
for f in mdsys.cgdir.iterdir():
    print(f)
#%% 
# These two commands will combine all topology and structure files. "make_cg_structure" uses GROMACS "gmx pdb2gmx" 
# module to make the simulation box, and its description can be found online 
mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
mdsys.make_cg_structure(bt='dodecahedron', d='1.2', ) # CG structure. Returns mdsys.solupdb ("solute.pdb") file    

#%% 
# Now we need to add solvent and neutralize the system's charge
solvent = mdsys.root / "water.gro"
mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")

#%% 
# In order to work with GROMACS' selections we need to make a .ndx file mdsys.sysndx.
# Order of the groups: 1.System,  2.Solute,  3.Backbone,  4.Solvent,  5. Not Water,  6-. Chains.
# Custom groups can be added using AtomList.write_to_ndx() method
mdsys.make_system_ndx(backbone_atoms=["BB"])

#%%
# One of the little quality of life features of GmxSystem is executing GROMACS commands from your python script
# with GmxSystem.gmx. This will run the command in the working directory of the system.
# To take a look at the system, we can use VMD or PyMol, but we need to correct the generated box for
# the periodic boundary conditions, and we can do it with GROMACS' "trjconv" module. 
mdsys.gmx("trjconv", clinput='0\n', s=mdsys.syspdb, f=mdsys.syspdb, pbc='atom', ur='compact', o="viz.pdb")
