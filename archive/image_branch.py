import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchio as tio

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
        """Normalize the image by subtracting mean and dividing by std."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:  # To avoid division by zero
            std = 1.0
        return (image - mean) / std

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
        image = np.stack([t1, t1ce, flair, t2, seg], axis=0)

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

class GaussianNoise3D(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise3D, self).__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        if self.training:  # Apply noise only during training
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x
    
class encoder(nn.Module):
    def __init__(self, input_shape, network_depth, no_convolutions, conv_filter_no_init, 
                 conv_kernel_size, latent_representation_dim, l1, l2, dropout_value, 
                 use_batch_normalization, activation, gaussian_noise_std=None):
        super(encoder, self).__init__()
        self.input_shape = input_shape
        self.network_depth = network_depth
        self.no_convolutions = no_convolutions
        self.conv_filter_no_init = conv_filter_no_init
        self.conv_kernel_size = conv_kernel_size
        self.latent_representation_dim = latent_representation_dim
        self.l1 = l1
        self.l2 = l2
        self.dropout_value = dropout_value
        self.use_batch_normalization = use_batch_normalization
        self.activation = activation
        self.gaussian_noise_std = gaussian_noise_std   
        self.encoder_layers = nn.ModuleList()

        # Gaussian noise layer
        if gaussian_noise_std:
            self.noise_layer = GaussianNoise3D(gaussian_noise_std)
        else:
            self.noise_layer = None

        # Convolutional layers
        in_channels = input_shape[0]
        for i in range(network_depth):
            for j in range(no_convolutions):
                out_channels = self.conv_filter_no_init * (2 ** i)
                conv_layer = nn.Conv3d(in_channels, out_channels, conv_kernel_size, padding=conv_kernel_size // 2)
                self.encoder_layers.append(conv_layer)
                if self.use_batch_normalization:
                    self.encoder_layers.append(nn.BatchNorm3d(out_channels))
                if self.activation == 'leakyrelu':
                    self.encoder_layers.append(nn.LeakyReLU(inplace=True))
                else:
                    self.encoder_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            self.encoder_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            if dropout_value:
                self.encoder_layers.append(nn.Dropout3d(p=dropout_value))

        self.flatten = nn.Flatten()

        # Calculate feature map size after convolution
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self._forward_conv_layers(dummy_input)
            self.feature_map_size = conv_output.size()
            flattened_dim = conv_output.view(1, -1).size(1)

        # Fully connected layer
        self.fc = nn.Linear(flattened_dim, latent_representation_dim)
        if activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(inplace=True)
        else:
            self.activation_fn = nn.ReLU(inplace=True)
            
    def _forward_conv_layers(self, x):
        if self.noise_layer:
            x = self.noise_layer(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation_fn(x)
        return x
    
def downsample_fn(depth):
    if depth == 4:
        return [(2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2)]
    elif depth == 2:
        return [(4, 4, 1), (4, 4, 4)]
    else:
        raise ValueError(f'Unsupported depth: {depth}')
    
class Decoder3D(nn.Module):
    def __init__(self, conv_shape, network_depth, no_convolutions, conv_filter_no_init,
                 conv_kernel_size, latent_representation_dim, l1=0.0, l2=0.0,
                 dropout_value=0.0, use_batch_normalization=False, activation='relu'):
        super(Decoder3D, self).__init__()
        self.conv_shape = conv_shape
        self.network_depth = network_depth
        self.no_convolutions = no_convolutions
        self.conv_filter_no_init = conv_filter_no_init
        self.conv_kernel_size = conv_kernel_size
        self.latent_representation_dim = latent_representation_dim
        self.l1 = l1
        self.l2 = l2
        self.dropout_value = dropout_value
        self.use_batch_normalization = use_batch_normalization
        self.activation = activation
        if activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(inplace=True)
        else:
            self.activation_fn = nn.ReLU(inplace=True)

        # Reshape layer
        self.reshape = lambda x: x.view(-1, *conv_shape)
        self.decoder_layers = nn.ModuleList()
        # Add layers in reverse according to the network depth
        for i in reversed(range(network_depth)):
            # Upsampling layer
            upsample_factors = downsample_fn(network_depth)[i]
            self.decoder_layers.append(nn.Upsample(scale_factor=upsample_factors, mode='trilinear', align_corners=False))
            
            # Convolution layers
            in_channels = self.conv_filter_no_init * (2 ** (i+1))
            out_channels = self.conv_filter_no_init * (2 ** i)
            for j in range(no_convolutions):
                self.decoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding='same'))
                if use_batch_normalization:
                    self.decoder_layers.append(nn.BatchNorm3d(out_channels))
                if activation == 'leakyrelu':
                    self.decoder_layers.append(nn.LeakyReLU(inplace=True))
                else:
                    self.decoder_layers.append(self.activation_fn)
                in_channels = out_channels
            if dropout_value > 0.0:
                self.decoder_layers.append(nn.Dropout3d(p=dropout_value))
        self.final_conv = nn.Conv3d(conv_shape[0], conv_shape[0], conv_kernel_size, padding='same') #change this 
        self.final_activation = nn.ReLU()

    def forward(self, x):
        # Expand the latent vector
        x = self.fc(x)
        x = self.activation_fn(x)
        
        # Reshape to the shape required for the convolutional layers
        x = self.reshape(x)
        
        # Apply the decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final convolution to produce the output volume
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x
    
def downsample_fn(depth):
    if depth == 4:
        return [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]  # Increase depth upsample factor to 2
    elif depth == 2:
        return [(4, 4, 4), (4, 4, 4)]  # Increase depth upsample factor to 4 if needed
    else:
        raise ValueError(f'Unsupported depth: {depth}')
    
class Decoder3D(nn.Module):
    def __init__(self, conv_shape, network_depth, no_convolutions, conv_filter_no_init,
                 conv_kernel_size, latent_representation_dim, output_channels=5, l1=0.0, l2=0.0,
                 dropout_value=0.0, use_batch_normalization=False, activation='relu'):
        super(Decoder3D, self).__init__()
        self.conv_shape = conv_shape  # Shape of the feature map at the start of the decoder
        self.network_depth = network_depth
        self.no_convolutions = no_convolutions
        self.conv_filter_no_init = conv_filter_no_init
        self.conv_kernel_size = conv_kernel_size
        self.latent_representation_dim = latent_representation_dim
        self.l1 = l1
        self.l2 = l2
        self.dropout_value = dropout_value
        self.use_batch_normalization = use_batch_normalization
        self.activation = activation
        self.output_channels = output_channels  # Final output channels (e.g., 5 channels for MRI modalities)
        
        # Activation function
        if activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(inplace=True)
        else:
            self.activation_fn = nn.ReLU(inplace=True)

        # Fully connected layer to reshape the latent vector into a 3D shape
        self.fc = nn.Linear(latent_representation_dim, np.prod(self.conv_shape))

        # Reshape layer to convert the flat output of the FC layer into a 3D volume
        self.reshape = lambda x: x.view(-1, *self.conv_shape)
        
        # Decoder layers (upsample and conv layers)
        self.decoder_layers = nn.ModuleList()
        
        # Reverse the depth, so we progressively upsample back to the original image size
        in_channels = conv_shape[0]  # Start with the number of channels from the conv_shape
        for i in reversed(range(network_depth)):
            # Upsampling layer
            upsample_factors = downsample_fn(network_depth)[i]
            self.decoder_layers.append(nn.Upsample(scale_factor=upsample_factors, mode='trilinear', align_corners=False))
            
            # Convolution layers
            out_channels = self.conv_filter_no_init * (2 ** i)  # Reduce the number of channels as we move up the network
            for j in range(no_convolutions):
                print(f"Layer {i}-{j}: in_channels = {in_channels}, out_channels = {out_channels}")
                self.decoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=1))
                if use_batch_normalization:
                    self.decoder_layers.append(nn.BatchNorm3d(out_channels))
                if activation == 'leakyrelu':
                    self.decoder_layers.append(nn.LeakyReLU(inplace=True))
                else:
                    self.decoder_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels  # Update in_channels for the next layer
            if dropout_value > 0.0:
                self.decoder_layers.append(nn.Dropout3d(p=dropout_value))
        
        # Final convolution to produce the reconstructed image with the correct number of output channels
        print(f"Final Layer: in_channels = {in_channels}, out_channels = {self.output_channels}")
        self.final_conv = nn.Conv3d(in_channels, self.output_channels, conv_kernel_size, padding=1)
        self.final_activation = nn.ReLU()  # You could change this to `nn.Sigmoid()` or `nn.Tanh()` depending on the data range

    def forward(self, x):
        # Expand the latent vector via the fully connected layer
        x = self.fc(x)
        x = self.activation_fn(x)
        
        # Reshape to the shape required for the convolutional layers
        x = self.reshape(x)  # Reshape to 3D tensor (batch_size, channels, depth, height, width)
        
        # Apply the decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final convolution to produce the output volume
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

class LatentParametersModel(nn.Module):
    def __init__(self, latent_representation_dim, l1=0.0, l2=0.0):
        super(LatentParametersModel, self).__init__()
        self.mu_sigma_layer = nn.Linear(
            in_features=latent_representation_dim, 
            out_features=2
        )
        nn.init.xavier_uniform_(self.mu_sigma_layer.weight)
        nn.init.zeros_(self.mu_sigma_layer.bias)
        self.l1 = l1
        self.l2 = l2
    
    def forward(self, x):
        mu_sigma = self.mu_sigma_layer(x)
        return mu_sigma
    

def reconstruction_loss(y_true, y_pred):
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    reduced_loss = mse_loss.mean(dim=[1, 2, 3])
    return reduced_loss

def survival_loss(mu, sigma, x, delta):
    """
    Custom loss function based on the negative log-likelihood.

    :param mu: Predicted mean (log of hazard ratio), tensor of shape (batch_size,)
    :param sigma: Predicted standard deviation (scale parameter), tensor of shape (batch_size,)
    :param x: Observed time (log-transformed), tensor of shape (batch_size,)
    :param delta: Event indicator (1 if event occurred, 0 if censored), tensor of shape (batch_size,)
    :return: Computed loss, scalar value
    """
    # Negative log-likelihood term
    total_loss = -(torch.log(x)-mu)/sigma.sum()+(delta * torch.log(sigma) + (1 + delta) * torch.log(1 + torch.exp((torch.log(x)-mu)/sigma)))
    
    # Return the mean loss across the batch
    return total_loss / x.size(0)




