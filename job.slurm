#!/bin/bash
#SBATCH -A m4505
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J skmtea-job
#SBATCH -o slurm-%j.out

# Move to your working directory
cd /global/homes/p/peterwg/pscratch/aujasvit/tackle

# Ensure conda is initialized properly in batch context
source ~/.bashrc
conda activate tackle-baseline

# Set Python path
export PYTHONPATH="$(pwd)"

# Run your Python script
srun python codesign/data/skmtea/skmtea.py
