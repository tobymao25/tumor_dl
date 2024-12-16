#!/usr/bin/env bash
#SBATCH --job-name=model_train_GBM
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=36G
#SBATCH --output=/projects/gbm_modeling/ltang35/tumor_dl/out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ltang35@jh.edu

# activate environment
source ~/.bashrc
conda activate monai_env

#python main.py --mode inference --model_path ./path/to/model_6_07_2024_batch1.pth --data_path ./path/to/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
python image_branch_main.py