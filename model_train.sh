#!/usr/bin/env bash
#SBATCH --job-name=model_train_GBM
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/ltang35/tumor_dl/out/%j.out

# activate environment
source ~/.bashrc
conda activate imgenv

python main.py --mode train --model_path ./path/to/model.pth --dataset_path ./path/to/dataset