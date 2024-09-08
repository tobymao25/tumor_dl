#!/usr/bin/env bash
#SBATCH --job-name=model_train_GBM
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10

#SBATCH --output=/projects/gbm_modeling/test/tumor_dl/out/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ymao17@jh.edu

# activate environment
source ~/.bashrc
conda activate glionet

#python main.py --mode inference --model_path ./path/to/model_6_07_2024_batch1.pth --data_path ./path/to/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
python /projects/gbm_modeling/test/tumor_dl/image_branch_main.py 