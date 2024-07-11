import sys
import torch
import argparse
from train import TrainTestPipe
from inference import SegInference
import seg_metrics.seg_metrics as sg
import os

def get_nii_gz_file_paths(root_dir):
    nii_gz_paths = []
    for file in os.listdir(root_dir):
        if file.endswith('.nii.gz'):
            full_path = os.path.join(root_dir, file)
            nii_gz_paths.append(full_path)
    return nii_gz_paths

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument('--dataset_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--data_path', required='infer' in sys.argv, type=str, default=None)
    parser = parser.parse_args()

    if parser.mode in ['train', 'evaluate']:
        assert parser.dataset_path is not None, 'dataset_path must be defined in training mode!'
    elif parser.mode == 'inference':
        assert parser.data_path is not None, 'data_path must be defined in inference mode!'
    
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
        pipeline = TrainTestPipe(mode="evaluate",
                                dataset_path=parser.dataset_path,
                                model_path=parser.model_path,
                                device=device)
        pipeline.evaluate()
    elif parser.mode == 'inference':
        pipeline = SegInference(model_path=parser.model_path,
                           device=device)
        '''
        #if single file implemented well
        img_path = get_nii_gz_file_paths(parser.data_path)
        for file in img_path:
            TC_path, WT_path, ET_path = pipeline.infer(file)
            print("now creating metrics for performance evaluation")
            labels = [0,1,2,3]
            for idx, i in enumerate([TC_path, WT_path, ET_path]):
                csv_file = 'metrics_{idx.csv' #add }
                metrics = sg.write_metrics(labels=labels[1:],  
                            gdth_path=parser.data_path,
                            pred_path=i,
                            csv_file=csv_file,
                            metrics=['dice', 'precision','hd95','msd'])'''
        TC_path, WT_path, ET_path = pipeline.save_masks(parser.data_path, pipeline.infer(parser.data_path))
    
    print("now creating metrics for performance evaluation")
    labels = [0,1,2,3]
    for idx, i in enumerate([TC_path, WT_path, ET_path]):
        csv_file = 'metrics_{idx}.csv'
        metrics = sg.write_metrics(labels=labels[1:],  
                    gdth_path=parser.data_path,
                    pred_path=i,
                    csv_file=csv_file,
                    metrics=['dice', 'precision','hd95','msd'])
        
        



    


    
