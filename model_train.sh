#!/usr/bin/env bash
#SBATCH --job-name=model_train_GBM
#SBATCH --partition=gpuq-a100
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --nodelist=mrphpcg012
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/ltang35/tumor_dl/out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ltang35@jh.edu

# activate environment
source ~/.bashrc
conda activate monai_env

#python main.py --mode inference --model_path ./path/to/model_6_07_2024_batch1.pth --data_path ./path/to/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
python image_branch_main.py