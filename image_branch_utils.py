import os
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd


class GBMdataset(Dataset):
    def __init__(self, image_dir, csv_path, target_dimensions=(128, 128, 128), target_spacing=(1, 1, 1), transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            csv_path (str): Path to the CSV file containing patient metadata.
            target_dimensions (tuple): Desired output image dimensions (e.g., 128x128x128).
            target_spacing (tuple): Target voxel spacing for resampling (e.g., 1x1x1 mm).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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
        return len(self.patient_ids)
    
    def _resample_image(self, image_path):
        """Resample the image to the target voxel spacing using torchio."""
        image = tio.ScalarImage(image_path)
        resample_transform = tio.transforms.Resample(self.target_spacing)
        resampled_image = resample_transform(image)
        return resampled_image
    
    def _resize_image(self, image):
        """Resize the image to the target dimensions (e.g., 128x128x128)."""
        resize_transform = tio.transforms.Resize(self.target_dimensions)
        resized_image = resize_transform(image)
        return resized_image.data.numpy()
    
    def _normalize_image(self, image):
        """Normalize each 2D slice along the depth of the 3D image to [0, 1]."""
        normalized_image = np.zeros_like(image)  # Create an array to hold the normalized image
        for i in range(image.shape[2]):  # Loop through each slice along the third axis (depth)
            slice_2d = image[:, :, i]  # Get the 2D slice
            min_val = np.min(slice_2d)
            max_val = np.max(slice_2d)
            
            # Normalize the slice to [0, 1], avoiding division by zero
            if max_val > min_val:
                normalized_image[:, :, i] = (slice_2d - min_val) / (max_val - min_val)
            else:
                normalized_image[:, :, i] = slice_2d  # If all values are the same, keep the original slice
        
        return normalized_image


    def __getitem__(self, idx):
        # Get the patient ID
        patient_id = self.patient_ids[idx]

        # Construct the file paths for the MRI images and segmentation
        t1_path = os.path.join(self.image_dir, f"{patient_id}_t1.nii")
        t1ce_path = os.path.join(self.image_dir, f"{patient_id}_t1ce.nii")
        flair_path = os.path.join(self.image_dir, f"{patient_id}_flair.nii")
        t2_path = os.path.join(self.image_dir, f"{patient_id}_t2.nii")
        seg_path = os.path.join(self.image_dir, f"{patient_id}_seg.nii")
        
        # Load and resample the images
        t1 = self._resample_image(t1_path)
        t1ce = self._resample_image(t1ce_path)
        flair = self._resample_image(flair_path)
        t2 = self._resample_image(t2_path)
        seg = self._resample_image(seg_path)

        # Resize the images to target dimensions (e.g., 128x128x128)
        t1 = self._resize_image(t1)
        t1ce = self._resize_image(t1ce)
        flair = self._resize_image(flair)
        t2 = self._resize_image(t2)
        seg = self._resize_image(seg)

        # Print the shape of the image after resampling and resizing
        #print(f"Image shape after resampling and resizing: {t1.shape}")
        
        # Normalize the images (for each modality)
        t1 = self._normalize_image(t1)
        t1ce = self._normalize_image(t1ce)
        flair = self._normalize_image(flair)
        t2 = self._normalize_image(t2)
        
        # Stack the images and segmentation into a single tensor
        image = np.stack([t1, seg], axis=0)
        image = torch.tensor(image, dtype=torch.float32).squeeze(1)
        # Print the size of the image after resampling, resizing, and normalization
        #print(f"Image shape after resampling, resizing, and normalization: {image.shape}")
        
        # Get the survival time for this patient
        survival_time = self.patient_data[patient_id]['Survival']
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Convert image and survival time to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        survival_time = torch.tensor(survival_time, dtype=torch.float32)
        
        return image, survival_time

#this is the modified Gaussian noise, could try to not use it or use it since it is correct now. 
class GaussianNoise(nn.Module):
    def __init__(self, noise_factor=0.1):
        super(GaussianNoise, self).__init__()
        self.noise_factor = noise_factor

    def forward(self, input_tensor):
        if self.training:
            std = torch.std(input_tensor)
            noise = torch.randn_like(input_tensor) * std * self.noise_factor
            noisy_tensor = input_tensor + noise  
            return noisy_tensor
        return input_tensor
