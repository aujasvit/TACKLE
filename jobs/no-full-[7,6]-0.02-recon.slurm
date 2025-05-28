#!/bin/bash
#SBATCH --job-name=NO-reconstruction-sweep#
#SBATCH --account=m4505
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --qos=debug
##SBATCH --qos=regular
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --output=/global/homes/p/peterwg/pscratch/aujasvit/slurm/no-full-run/logs/%j-%x.out

# Environment setup
source ~/.bashrc
cd /global/homes/p/peterwg/pscratch/aujasvit/tackle
export PYTHONPATH="$(pwd)"
conda activate tackle-baseline

export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_API_KEY=9bdf67e085edee3d5f0201ca102ccc6e0e9577a6
export WANDB_PROJECT=mri_codesign
export WANDB_ENTITY=aujasvitd
export WANDB_DIR=/global/homes/p/peterwg/pscratch/aujasvit/wandb
export WANDB_CACHE_DIR=/pscratch/sd/p/peterwg/more/wandb_cache



# Run script
# SWEEP_ID=aujasvitd/no_param_sweep/i6dh5huo
# SWEEP_ID=aujasvitd/no_param_sweep/tgwupoqu
# SWEEP_ID=aujasvitd/no_param_sweep/k1vmflq5


# wandb agent --count 16 $SWEEP_ID

srun python scripts/main-original.py \
  -c configs/custom/segmentation/novarnet/16x/16x_novarnet-recon_1e-3_2-cascade.yaml \
  --fix \
  exp_name novarnet-recon_2-cascade_ks[7,6]_rc0.02\
  model.reconstructor.kernel_shape "[7,6]" \
  model.reconstructor.radius_cutoff 0.02

 