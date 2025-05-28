#!/bin/bash
#SBATCH --job-name=NO-reconstruction-sweep#
#SBATCH --account=m4505
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --mem=150G
#SBATCH --time=10:00:00
##SBATCH --time=12:00:00
#SBATCH --output=/global/homes/p/peterwg/pscratch/aujasvit/slurm/no-recon-sweep/logs/%j-%x.out

# Environment setup
source ~/.bashrc
cd /global/homes/p/peterwg/pscratch/aujasvit/tackle
export PYTHONPATH="$(pwd)"
conda activate tackle-baseline

export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY=9bdf67e085edee3d5f0201ca102ccc6e0e9577a6
export WANDB_PROJECT=no_param_sweep
export WANDB_ENTITY=aujasvitd
export WANDB_DIR=/global/homes/p/peterwg/pscratch/aujasvit/wandb
export WANDB_CACHE_DIR=/pscratch/sd/p/peterwg/more/wandb_cache

# Run script
srun python train.py