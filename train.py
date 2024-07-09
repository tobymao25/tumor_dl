from tqdm import tqdm
from utils import EpochCallback, get_dataloader
from config import cfg
from train_unetr import UneTRSeg
import torch
import nibabel as nib
import os
import matplotlib.pyplot as plt
from monai.metrics import compute_hausdorff_distance
import numpy as np
class TrainTestPipe:
    def __init__(self, mode=None, dataset_path=None, model_path=None, device=None):
        self.device = device
        self.model_path = model_path

        if mode == "train":
            self.train_loader = get_dataloader(dataset_path, train=True)

        self.val_loader = get_dataloader(dataset_path, train=False)
        self.unetr = UneTRSeg(self.device)
        if torch.cuda.device_count() > 1:
            print(f"training using {torch.cuda.device_count()} GPUs!")
            self.unetr.model = torch.nn.DataParallel(self.unetr.model)

        self.unetr.model.to(self.device)

    def __loop(self, loader, step_func, t):
        total_loss = 0
        all_pred_masks = [] #change 
        all_labels = [] #change
        for step, data in enumerate(loader):
            image, label = data['image'], data['label']
            image = image.to(self.device)
            label = label.to(self.device)
            loss, pred_mask = step_func(image=image, label=label)
            all_pred_masks.extend([pm.cpu().detach().numpy() for pm in pred_mask]) #change
            all_labels.extend([lb.cpu().detach().numpy() for lb in label]) #change
            total_loss += loss
            t.update()

        return total_loss, all_pred_masks, all_labels #change

    def train(self):
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.unetr.model, self.unetr.optimizer, 'val_loss', cfg.patience)

        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.val_loader)) as t:
                train_loss = self.__loop(self.train_loader, self.unetr.train_step, t)

                val_loss = self.__loop(self.val_loader, self.unetr.val_step, t)

            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'val_loss': val_loss / len(self.val_loader)})

            if callback.end_training:
                break

        print("Evaluating...")
        self.evaluate()

    def evaluate(self):
        self.unetr.load_model(self.model_path)
        with tqdm(total=len(self.val_loader)) as t:
            val_loss, all_pred_masks, all_labels = self.__loop(self.val_loader, self.unetr.eval_step, t)
        dice_metric = self.unetr.metric.aggregate()
        print(f"TC Dice coefficient: {round(dice_metric[0].item(), 2)}")
        print(f"WT Dice coefficient: {round(dice_metric[1].item(), 2)}")
        print(f"ET Dice coefficient: {round(dice_metric[2].item(), 2)}")
        self.plot_bland_altman(all_pred_masks, all_labels)
        self.plot_hausdorff_distance(all_pred_masks, all_labels)
        self.save_nifti_files(all_pred_masks)
    
    def plot_bland_altman(self, pred_masks, gt_masks):
        pred_flat = np.concatenate([mask.flatten() for mask in pred_masks])
        gt_flat = np.concatenate([mask.flatten() for mask in gt_masks])

        mean = np.mean([pred_flat, gt_flat], axis=0)
        diff = pred_flat - gt_flat

        plt.figure(figsize=(10, 5))
        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(np.mean(diff), color='red', linestyle='--')
        plt.axhline(np.mean(diff) + 1.96 * np.std(diff), color='blue', linestyle='--')
        plt.axhline(np.mean(diff) - 1.96 * np.std(diff), color='blue', linestyle='--')
        plt.xlabel('Mean of Predicted and Ground Truth')
        plt.ylabel('Difference between Predicted and Ground Truth')
        plt.title('Bland-Altman Plot')
        plt.show()

    def plot_hausdorff_distance(self, pred_masks, gt_masks):
        hausdorff_distances = []
        for pred, gt in zip(pred_masks, gt_masks):
            hausdorff_dist = compute_hausdorff_distance(pred, gt)
            hausdorff_distances.append(hausdorff_dist)

        plt.figure(figsize=(10, 5))
        plt.plot(hausdorff_distances)
        plt.xlabel('Sample Index')
        plt.ylabel('Hausdorff Distance')
        plt.title('Hausdorff Distance Plot')
        plt.show()
        
    def save_nifti_files(self, pred_masks):
        os.makedirs('output/predicted_masks', exist_ok=True)

        for i, pred in enumerate(pred_masks):
            pred_nifti = nib.Nifti1Image(pred, np.eye(4))
            nib.save(pred_nifti, f'output/predicted_masks/pred_mask_{i}.nii.gz')

