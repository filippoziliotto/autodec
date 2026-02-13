#!/bin/sh

#SBATCH --job-name=PrepareABO
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

python superdec/data/prepare_abo.py