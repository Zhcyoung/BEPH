#!/bin/bash
#SBATCH --job-name=yzc
#SBATCH -J prep 
#SBATCH -p a100
#SBATCH -o prep.out
#SBATCH -e prep.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6


module load mmselfsup_yzc
checkpoint_file=$(cat /dssg/home/acct-medftn/medftn/BEPT/Model/mmselfsup/work_dirs/selfsup/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/last_checkpoint)

cd /dssg/home/acct-medftn/medftn/BEPT/Model/mmselfsup/
sh tools/slurm_train_8gpu.sh a100 ZhaochangYang_BEIT  ./TrainConfigs/beitv2_vit-moreGPU.py  -resume $checkpoint_file