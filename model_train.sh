#!/usr/bin/env bash
#SBATCH --job-name=model_train_GBM
#SBATCH --partition=gpuq-a100
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --nodelist=mrphpcg012
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/ltang35/tumor_dl/out/%j.out
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=ltang35@jhu.edu

# activate environment
source ~/.bashrc
conda activate imgenv

python main.py --mode train --model_path ./path/to/model.pth --dataset_path ./path/to/dataset