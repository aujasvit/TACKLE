#!/bin/bash
#SBATCH --job-name=NO-reconstruction-sweep-new#
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --account=m4505
#SBATCH --output=/global/homes/p/peterwg/pscratch/aujasvit/slurm/no-recon-sweep-new/logs/%j-%x.out

# Environment setup
source ~/.bashrc
cd /pscratch/sd/p/peterwg/aujasvit/tackle
export PYTHONPATH="$(pwd)"
conda activate tackle-baseline

export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY=9bdf67e085edee3d5f0201ca102ccc6e0e9577a6
export WANDB_PROJECT=no_param_sweep
export WANDB_ENTITY=aujasvitd
export WANDB_DIR=/pscratch/sd/p/peterwg/aujasvit/wandb
export WANDB_CACHE_DIR=/pscratch/sd/p/peterwg/more/wandb_cache

SWEEP_ID=aujasvitd/no_param_sweep/tgwupoqu

# Launch WANDB agent
srun wandb agent --count 1 $SWEEP_ID
