#!/bin/bash --login
#
#SBATCH -p gpuA              
#SBATCH -G 1                  
#SBATCH -n 8                  
#SBATCH -t 1-0               
#SBATCH -J AD_hmdb51              
#SBATCH -o logs/%x_%j.out     
#SBATCH -e logs/%x_%j.err      


module purge
module load libs/cuda/12.8.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate actionclip

export WANDB_API_KEY='enter your API key here'

cd 'enter your file root'

python train.py \
  --config configs/hmdb51/hmdb_256.yaml \
  --log_time 5shot_adapter_8 \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S)_5shot_adapter_8.log
