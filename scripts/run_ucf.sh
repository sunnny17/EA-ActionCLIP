#!/bin/bash --login
#
#SBATCH -J AD_clip_ucf                
#SBATCH -p gpuA                        
#SBATCH -G 1                          
#SBATCH -n 8                           
#SBATCH -t 2-0                         
#SBATCH -o logs/ucf_train_%j.out       
#SBATCH -e logs/ucf_train_%j.err      
  
  
module purge
module load libs/cuda/12.8.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate actionclip

export WANDB_API_KEY='enter your API key number'

cd 'enter your file root'

python train.py \
  --config configs/ucf101/ucf_few_train7.yaml \
  --log_time 5shot_adapter_6 \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S)_5shot_adapter_6.log
