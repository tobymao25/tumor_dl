import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from image_branch_utils import GaussianNoise


class encoder(nn.Module):
    def __init__(self, input_shape, network_depth, no_convolutions, conv_filter_no_init, 
                 conv_kernel_size, latent_representation_dim, l1, l2, dropout_value, 
                 use_batch_normalization, activation, gaussian_noise_factor=0.3):
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
        self.encoder_layers = nn.ModuleList()
        self.gaussian_noise_factor = gaussian_noise_factor
        
        # Gaussian noise layer
        if gaussian_noise_factor:
            self.noise_layer = GaussianNoise()
        else:
            self.noise_layer = None

        # Convolutional layers
        in_channels = self.input_shape[0]
        for i in range(network_depth):
            for j in range(no_convolutions):
                out_channels = self.conv_filter_no_init * (2 ** i)
                conv_layer = nn.Conv3d(in_channels, out_channels, self.conv_kernel_size, padding=(conv_kernel_size-1) // 2)
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
            dummy_input = torch.zeros(1, *self.input_shape)
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
                self.decoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size-1) // 2))
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
        self.final_conv = nn.Conv3d(in_channels, self.output_channels, conv_kernel_size, padding=(conv_kernel_size-1) // 2)
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
        mu_sigma = loglogistic_activation(mu_sigma)
        return mu_sigma


def reconstruction_loss(y_true, y_pred):
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    reduced_loss = mse_loss.mean(dim=[1, 2, 3])
    return reduced_loss


def survival_loss(mu, logsigma, x, delta):
    """
    Custom loss function based on the negative log-likelihood.

    :param mu: Predicted mean (log of hazard ratio), tensor of shape (batch_size,)
    :param sigma: Predicted standard deviation (scale parameter), tensor of shape (batch_size,)
    :param x: Observed time (log-transformed), tensor of shape (batch_size,)
    :param delta: Event indicator (1 if event occurred, 0 if censored), tensor of shape (batch_size,)
    :return: Computed loss, scalar value
    """
    x_scaled = (torch.log(x) - mu) / torch.exp(logsigma)
    nll = torch.sum(x_scaled + delta * logsigma + (1 + delta) * torch.log(1 + torch.exp(-x_scaled)))

    # calculate MSE for evaluating model
    MSE_loss = nn.MSELoss()
    MSE = MSE_loss(torch.exp(mu), x)

    return nll, MSE #no need to negate and exponentiate 

def loglogistic_activation(mu_logsig):
    """
    Activation which ensures mu is between -3 and 3 and sigma is such that
    prediction is not more precise than 1 / n of a year.
    :param mu_logsig: Tensor containing [mu, log(sigma)]
    :return: Tensor with updated mu and log(sigma)
    """
    n = 365  # 1 / n is the fraction of the year in which at least p quantile of the distribution lies
    p = 0.95  # quantile

    # Clip mu between the min and max survival from brats data set, change this in the future
    mu = torch.clamp(mu_logsig[:, 0], -3, 3) #0.6931, 7.4593)

    # Calculate sigma by exponentiating the second column
    sig = torch.exp(mu_logsig[:, 1])

    # Threshold calculation based on the given formula
    thrs = torch.log((1 / (2 * n)) * (torch.exp(-mu) + torch.sqrt((2 * n) ** 2 + torch.exp(-2 * mu)))) / \
           torch.log(torch.tensor(p / (1 - p), dtype=torch.float32))

    # Ensure sigma is no more precise than the threshold
    logsig = torch.log(thrs + F.relu(sig - thrs))

    # Reshape mu and logsig into column vectors
    mu = mu.view(-1, 1)
    logsig = logsig.view(-1, 1)

    # Concatenate mu and logsig along the last axis
    new = torch.cat((mu, logsig), dim=1)
    
    return new

class GlioNet(nn.Module):
    def __init__(self, encoder, decoder, latent_param_model):
        super(GlioNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_param_model = latent_param_model

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstruction = self.decoder(latent_representation)
        latent_params = self.latent_param_model(latent_representation)
        return reconstruction, latent_params

