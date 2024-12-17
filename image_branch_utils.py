import os
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

"""This code includes a customized dataset that takes patient clinical covariate, MRI images, and survival time
as well as augmentation method, written by Yuncong Mao in August 2024"""

class GBMdataset(Dataset):
    def __init__(self, image_dir, csv_path, target_dimensions=(128, 128, 128), target_spacing=(1, 1, 1), transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.patient_data = self._load_patient_data(csv_path)
        self.patient_ids = list(self.patient_data.keys())
        self.target_dimensions = target_dimensions
        self.target_spacing = target_spacing

    # def _load_patient_data(self, csv_path): #### ! For pretrained model w/o covariates
    #     data = pd.read_csv(csv_path)
    #     patient_data = {}
    #     for _, row in data.iterrows():
    #         patient_id = row['Brats17ID']
    #         survival = row['Survival']
    #         patient_data[patient_id] = {'Survival': survival}
    #     return patient_data

    def _load_patient_data(self, csv_path):
        data = pd.read_csv(csv_path)
        patient_data = {}
        for _, row in data.iterrows():
            patient_id = row['ID'] #!
            survival = row['Survival']
            #cov = row.drop(['ID', 'Survival', 'Original Accession', 'MRN', "MRI_immediate_pre", "binned_outcome"]).values #!
            cov = row[["MFI", "KPS score","Age"]].values
            cov = pd.to_numeric(cov, errors='coerce')
            patient_data[patient_id] = {'Survival': survival, 'Covariates': cov}
        return patient_data  

    def __len__(self):
        # Each image has 4 versions: original, horizontal flip, vertical flip, and rotation
        return len(self.patient_ids) * 4

    def _resample_image(self, image_path, dtype):
        image = tio.ScalarImage(image_path)
        if dtype == "img": 
            resample_transform = tio.transforms.Resample(self.target_spacing, image_interpolation='linear')
        elif dtype == "seg": 
            resample_transform = tio.transforms.Resample(self.target_spacing, image_interpolation='nearest')
        else: 
            raise TypeError("Unsupported data type")
        resampled_image = resample_transform(image)
        return resampled_image

    def _resize_image(self, image, dtype):
        if dtype == "img": 
            resize_transform = tio.transforms.Resize(self.target_dimensions, image_interpolation='linear')
        elif dtype == "seg": 
            resize_transform = tio.transforms.Resize(self.target_dimensions, image_interpolation='nearest')
        else: 
            raise TypeError("Unsupported data type")
        resized_image = resize_transform(image)
        return resized_image.data.numpy()

    def _normalize_image(self, image):
        perc_9999_val = np.percentile(image, 99.99)
        min_val = np.min(image)
        normalized_image = (image - min_val) / (perc_9999_val - min_val + 1e-5)
        normalized_image = np.clip(normalized_image, a_min=0, a_max=1)
        return normalized_image

    def __getitem__(self, idx):
        # Determine the base patient index and augmentation type
        patient_idx = idx // 4  # Base patient index
        aug_idx = idx % 4       # 0 = original, 1 = horizontal flip, 2 = vertical flip, 3 = rotation

        # Get the patient ID
        patient_id = self.patient_ids[patient_idx]

        # Define augmentation suffix based on aug_idx
        if aug_idx == 0:
            aug_suffix = ''  # Original image has no suffix
        elif aug_idx == 1:
            aug_suffix = '_hf'  # Horizontal flip
        elif aug_idx == 2:
            aug_suffix = '_vf'  # Vertical flip
        elif aug_idx == 3:
            aug_suffix = '_r'  # Rotation
        else:
            raise ValueError("Invalid augmentation index")

        # Construct file paths with appropriate suffixes
        #t1_path = os.path.join(self.image_dir, f"{patient_id}_t1{aug_suffix}.nii")
        t1ce_path = os.path.join(self.image_dir, f"{patient_id}_t1ce{aug_suffix}.nii")
        flair_path = os.path.join(self.image_dir, f"{patient_id}_flair{aug_suffix}.nii")
        t2_path = os.path.join(self.image_dir, f"{patient_id}_t2{aug_suffix}.nii")
        #seg_path = os.path.join(self.image_dir, f"{patient_id}_seg{aug_suffix}.nii")
        
        # Load and resample the images
        #t1 = self._resample_image(t1_path, "img")
        t1ce = self._resample_image(t1ce_path, "img")
        flair = self._resample_image(flair_path, "img")
        t2 = self._resample_image(t2_path, "img")
        #seg = self._resample_image(seg_path, "seg")

        # Resize the images to target dimensions (e.g., 128x128x128)
        #t1 = self._resize_image(t1, "img")
        t1ce = self._resize_image(t1ce, "img")
        flair = self._resize_image(flair, "img")
        t2 = self._resize_image(t2, "img")
        #seg = self._resize_image(seg, "seg")

        # Standardize the images
        #t1 = self._normalize_image(t1)
        t1ce = self._normalize_image(t1ce)
        flair = self._normalize_image(flair)
        t2 = self._normalize_image(t2)

        # Normalize segmentation 
        # seg_min = np.min(seg)
        # seg_max = np.max(seg)
        # if seg_max != seg_min:
        #     seg = (seg - seg_min) / (seg_max - seg_min)

        # Stack the images and segmentation into a single tensor
        image = np.stack([t1ce, flair, t2], axis=0) #np.stack([t1, t1ce, flair, t2, seg], axis=0)
        
        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        # Get the survival time for this patient
        survival_time = self.patient_data[patient_id]['Survival']
        survival_time = torch.tensor(survival_time, dtype=torch.float32)

        #return image, survival_time #! pretrained model w/o cov

        # Get the covariate for this patient
        covariate = self.patient_data[patient_id]['Covariates']
        covariate = torch.tensor(covariate, dtype=torch.float32)

        return image, survival_time, covariate

# this is the modified Gaussian noise, could try to not use it or use it since it is correct now. 
class GaussianNoise(nn.Module):
    def __init__(self, noise_factor=0.05):
        super(GaussianNoise, self).__init__()
        self.noise_factor = noise_factor

    def forward(self, input_tensor):
        if self.training:
            std = torch.std(input_tensor)
            noise = torch.randn_like(input_tensor) * std * self.noise_factor
            # if using normalizaiton, uncomment this line to make sure no negative noise is added to range [0-1]
            noise = torch.abs(noise) 
            noisy_tensor = input_tensor + noise  
            return noisy_tensor
        return input_tensor


def augment_and_save(image_path, save_dir, dtype):
    """this function reads all the images in image_path folder and apply augmentation 
    so each original image will have one horizontally flipped, one vertically flipped,
    and one rotated image in the training dataset
    """
    image = tio.ScalarImage(image_path)
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    if dtype=="seg": 
        augmentations = {
        'hf': tio.transforms.Flip(axes=('LR',)),       # Horizontal flip (left-right)
        'vf': tio.transforms.Flip(axes=('AP',)),       # Vertical flip (anterior-posterior)
        'r': tio.transforms.RandomAffine(degrees=(30, 30, 30, 30, 30, 30), image_interpolation='nearest') # Small random rotation
        }
    elif dtype=="img": 
        augmentations = {
        'hf': tio.transforms.Flip(axes=('LR',)),       # Horizontal flip (left-right)
        'vf': tio.transforms.Flip(axes=('AP',)),       # Vertical flip (anterior-posterior)
        'r': tio.transforms.RandomAffine(degrees=(30, 30, 30, 30, 30, 30), image_interpolation='linear') # Small random rotation
        }
    else: 
        raise TypeError("Unsupported data type")
    for aug_suffix, transform in augmentations.items():
        augmented_image = transform(image)
        new_filename = f"{base_name}_{aug_suffix}{ext}"
        save_path = os.path.join(save_dir, new_filename)
        augmented_image.save(save_path)
        print(f"Saved: {save_path}")


def save(image_dir):
    """this is the helper function that calls the augment_and_save function that keeps track of the progress
    """
    for file_name in tqdm(os.listdir(image_dir)):
        if file_name.endswith(".nii"):
            file_path = os.path.join(image_dir, file_name)
            if file_name.endswith("seg.nii"):
                augment_and_save(file_path, image_dir, "seg")
            else:
                augment_and_save(file_path, image_dir, "img")
    print("Augmentation complete!")


if __name__ =='__main__':
    print("starting augmentation")
    save("/projects/gbm_modeling/ltang35/tumor_dl/TrainingDataset/Brats2020_unique")
    print("augmentation finished")
