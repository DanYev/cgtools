#!/usr/bin/env python
"""
Hello world!
=================

Interface tutorial.

Requirements:
    - HPC cluster
    - Python 3.x

Author: DY
"""
import os

WDIR = 'docs/examples' # '.' for html, 'examples' for manual
os.chdir(WDIR)
#%%
# One of the motivations behind reForge was to have a user- and beginner-friendly interface for managing 
# potentially hundreds or thousands of MD simulations without having to jump between multiple 
# scripts and constanly rewrite them. And without leaving the comfort of Python :-)
# That's what 'cli' module is for.
from reforge.cli import run, sbatch, dojob

#%%
# Imagine you have ten mutants and need to run ten independent runs for each of them to get enough sampling.
systems = ['mutant_' + i for i in range(10)]
mdruns = ['mdrun_' + i for i in range(10)]
print(systems)
print(mdruns)
