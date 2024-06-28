import sys
import torch
import argparse
from train import TrainTestPipe

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument('--dataset_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--data_path', required='infer' in sys.argv, type=str, default=None)
    parser = parser.parse_args()

    if parser.mode in ['train', 'evaluate']:
        assert parser.dataset_path is not None, 'dataset_path must be defined in training mode!'
    torch.cuda.empty_cache()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if parser.mode == 'train':
        pipeline = TrainTestPipe(mode="train",
                                dataset_path=parser.dataset_path,
                                model_path=parser.model_path,
                                device=device)
        pipeline.train()
    elif parser.mode == 'evaluate':
        pipline = TrainTestPipe(mode="evaluate",
                                dataset_path=parser.dataset_path,
                                model_path=parser.model_path,
                                device=device)
        pipline.evaluate()
