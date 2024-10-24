#!/bin/bash
#SBATCH --time=0-01:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -o slurm_output/output.%A.%a.out
#SBATCH -e slurm_output/error.%A.%a.err

PYSCRIPT="$@"   

python $PYSCRIPT
