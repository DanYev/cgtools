#!/bin/bash
#SBATCH --time=0-04:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=general
#SBATCH --qos=grp_sozkan                                                     # public grp_sozkan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
#SBATCH --gres=gpu:1	                                                        # number of GPUs
#SBATCH -o slurm_output/output.%A.%a.out
#SBATCH -e slurm_output/error.%A.%a.err

python cgtools/run_all.py