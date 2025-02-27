#!/bin/bash

find -name '#*' -delete
find -name 'step*.pdb' -delete
rm slurm_jobs/* > /dev/null 2>&1
