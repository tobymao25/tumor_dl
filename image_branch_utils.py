import os
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd


class GBMdataset(Dataset):
    def __init__(self, image_dir, csv_path, target_dimensions=(128, 128, 128), target_spacing=(1, 1, 1), transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.patient_data = self._load_patient_data(csv_path)
        self.patient_ids = list(self.patient_data.keys())
        self.target_dimensions = target_dimensions
        self.target_spacing = target_spacing

    def _load_patient_data(self, csv_path):
        data = pd.read_csv(csv_path)
        patient_data = {}
        for _, row in data.iterrows():
            patient_id = row['Brats17ID']
            age = row['Age']
            survival = row['Survival']
            patient_data[patient_id] = {'Age': age, 'Survival': survival}
        return patient_data
    
    def __len__(self):
        # Each image will have 3 versions: original, horizontally flipped, vertically flipped
        return len(self.patient_ids) * 3
    
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

    def _standardize_image(self, image):
        mean = np.mean(image)
        std = np.std(image)
        standardized_image = (image - mean) / (std + 1e-5)
        return standardized_image
    
    def _normalize_image(self, image):
        perc_9999_val = np.percentile(image, 99.99)
        min_val = np.min(image)
        normalized_image = (image - min_val) / (perc_9999_val - min_val + 1e-5)
        normalized_image = np.clip(normalized_image, a_min = 0, a_max = 1)
        return normalized_image

    def __getitem__(self, idx):
        # Get the base patient index (before augmentation) and mod_idx to determine augmentation
        patient_idx = idx // 3  # Original patient index
        mod_idx = idx % 3       # Determines which version (original, horizontal, vertical)

        # Get the patient ID
        patient_id = self.patient_ids[patient_idx]

        # Construct the file paths for the MRI images and segmentation
        t1_path = os.path.join(self.image_dir, f"{patient_id}_t1.nii")
        t1ce_path = os.path.join(self.image_dir, f"{patient_id}_t1ce.nii")
        flair_path = os.path.join(self.image_dir, f"{patient_id}_flair.nii")
        t2_path = os.path.join(self.image_dir, f"{patient_id}_t2.nii")
        seg_path = os.path.join(self.image_dir, f"{patient_id}_seg.nii")
        
        # Load and resample the images
        t1 = self._resample_image(t1_path, "img")
        t1ce = self._resample_image(t1ce_path, "img")
        flair = self._resample_image(flair_path, "img")
        t2 = self._resample_image(t2_path, "img")
        seg = self._resample_image(seg_path, "seg")

        # Resize the images to target dimensions (e.g., 128x128x128)
        t1 = self._resize_image(t1, "img")
        t1ce = self._resize_image(t1ce, "img")
        flair = self._resize_image(flair, "img")
        t2 = self._resize_image(t2, "img")
        seg = self._resize_image(seg, "seg")

        # Standardize the images
        t1 = self._normalize_image(t1)
        t1ce = self._normalize_image(t1ce)
        flair = self._normalize_image(flair)
        t2 = self._normalize_image(t2)

        # Normalize segmentation 
        seg_min = np.min(seg)
        seg_max = np.max(seg)
        if seg_max != seg_min:
            seg = (seg - seg_min) / (seg_max - seg_min)

        # Stack the images and segmentation into a single tensor
        image = np.stack([t1, t1ce, flair, t2, seg], axis=0)

        # Apply horizontal or vertical flip based on mod_idx
        if mod_idx == 1:
            # Horizontally flip the image and make a copy
            image = np.flip(image, axis=2).copy()  # Flip along the x-axis
        elif mod_idx == 2:
            # Vertically flip the image and make a copy
            image = np.flip(image, axis=3).copy()  # Flip along the y-axis
        
        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        # Get the survival time for this patient
        survival_time = self.patient_data[patient_id]['Survival']
        survival_time = torch.tensor(survival_time, dtype=torch.float32)

        return image, survival_time

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
