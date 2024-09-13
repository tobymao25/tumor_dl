import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from image_branch_utils import GaussianNoise

def downsample_fn(depth):
    if depth == 6:
        return [(2, 2, 2)] * 6 
    elif depth == 8:
        return [(1.5, 1.5, 1.5)] * 8 
    elif depth == 4:
        return [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]  # Increase depth upsample factor to 2
    elif depth == 2:
        return [(4, 4, 4), (4, 4, 4)]  # Increase depth upsample factor to 4 if needed
    else:
        raise ValueError(f'Unsupported depth: {depth}')


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
        self.skip_connections = []

        # Gaussian noise layer
        if gaussian_noise_factor:
            self.noise_layer = GaussianNoise()
        else:
            self.noise_layer = None

        # Convolutional layers
        in_channels = self.input_shape[0]
        for i in range(network_depth):
            conv_layers = []
            for j in range(no_convolutions):
                out_channels = self.conv_filter_no_init * (2 ** i) 
                conv_layer = nn.Conv3d(in_channels, out_channels, self.conv_kernel_size, padding=(conv_kernel_size-1) // 2)
                conv_layers.append(conv_layer)
                if self.use_batch_normalization:
                    conv_layers.append(nn.BatchNorm3d(out_channels))
                if self.activation == 'leakyrelu':
                    conv_layers.append(nn.LeakyReLU(inplace=True))
                else:
                    conv_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            
            self.encoder_layers.extend(conv_layers)
            self.encoder_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.skip_connections.append(conv_layers[-1])  

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
        
        skip_connections = []  

        for layer in self.encoder_layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool3d): 
                skip_connections.append(x.clone())

        self.skip_connections = skip_connections  
        return x
    
    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        print(f'this is the shape of fully connected layer: {x.shape}')
        x = self.activation_fn(x)
        return x


class Decoder3D(nn.Module):
    def __init__(self, conv_shape, network_depth, no_convolutions, conv_filter_no_init,
                 conv_kernel_size, latent_representation_dim, output_channels=5, l1=0.0, l2=0.0,
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
        self.output_channels = output_channels
        
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
        in_channels = conv_shape[0]
        
        for i in reversed(range(network_depth)):
            # Upsampling layer
            upsample_factors = downsample_fn(network_depth)[i]
            self.decoder_layers.append(nn.Upsample(scale_factor=upsample_factors, mode='trilinear', align_corners=False))
            
            # First convolution layer after concatenation needs to handle in_channels + out_channels channels
            out_channels = self.conv_filter_no_init * (2 ** i)
            self.decoder_layers.append(nn.Conv3d(in_channels+out_channels, out_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size-1) // 2))
            in_channels = out_channels
            for j in range(1, no_convolutions):
                self.decoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=(conv_kernel_size-1) // 2))
                
            
        
        self.final_conv = nn.Conv3d(in_channels, self.output_channels, conv_kernel_size, padding=(conv_kernel_size-1) // 2)
        self.final_activation = nn.ReLU()

    def forward(self, x, skip):
        x = self.fc(x)
        x = self.activation_fn(x)
        x = self.reshape(x)
        
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Upsample):
                skip_connection = skip.pop()
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                x = torch.cat((x, skip_connection), dim=1)
            x = layer(x)
            # if isinstance(layer, nn.Conv3d):
            #     print(f"Shape after Conv3D {i}: {x.shape}, expected input: {layer.weight.shape[1]}, expected output: {layer.weight.shape[0]}")
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x


class LatentParametersModel(nn.Module):
    def __init__(self, latent_representation_dim, l1=0.0, l2=0.0):
        super(LatentParametersModel, self).__init__()
        self.mu_layer = nn.Linear(
            in_features=latent_representation_dim, 
            out_features=1
        )
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        self.l1 = l1
        self.l2 = l2
    
    def forward(self, x):
        mu_sigma = self.mu_layer(x)
        #mu_sigma = loglogistic_activation(mu_sigma)
        return mu_sigma


def reconstruction_loss(y_true, y_pred):
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    reduced_loss = mse_loss.mean(dim=[1, 2, 3])
    return reduced_loss


def survival_loss(mu, x):
    """
    Custom loss function based on the negative log-likelihood.

    :param mu: Predicted mean (log of hazard ratio), tensor of shape (batch_size,)
    :param sigma: Predicted standard deviation (scale parameter), tensor of shape (batch_size,)
    :param x: Observed time (log-transformed), tensor of shape (batch_size,)
    :param delta: Event indicator (1 if event occurred, 0 if censored), tensor of shape (batch_size,)
    :return: Computed loss, scalar value
    """
    print("these are mu")
    print(mu)
    print("there are the labels, x")
    print(x)

    # New loss 1
    # sigma = torch.exp(logsigma)
    # sigma = sigma.clamp(min=0.001)
    # mu = torch.exp(mu)
    # # Calculate the NLL loss
    # log_x = torch.log(x + 0.001)  # Add epsilon to prevent log(0)
    # nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + 0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
    # #(------- trying out another loss function)

    # # New loss 2
    # # using the automatic implementation torch.nn.GaussianNLLLoss
    # nll_loss = nn.GaussianNLLLoss(eps=1e-06, reduction="mean")
    # nll = nll_loss(torch.exp(mu), x, torch.exp(logsigma))
    """
    sigma = torch.clamp(torch.exp(logsigma), min=0.1) #clamp to prevent gradient explosion
    x_scaled = (torch.log(x) - mu) / sigma 
    nll = torch.sum(x_scaled + delta * logsigma + (1 - delta) * torch.log(1 + torch.exp(-x_scaled)))"""

    # calculate MSE for evaluating model
    MSE_loss = nn.MSELoss(reduction='mean')
    MSE = MSE_loss(mu, x)

    return MSE

"""
def loglogistic_activation(mu_logsig):
    
    Activation which ensures mu is between -3 and 3 and sigma is such that
    prediction is not more precise than 1 / n of a year.
    :param mu_logsig: Tensor containing [mu, log(sigma)]
    :return: Tensor with updated mu and log(sigma)
    
    n = 1  # 1 / n is the fraction of the year in which at least p quantile of the distribution lies
    p = 0.95  # quantile

    # Clip mu between the min and max survival from brats data set, change this in the future
    mu = torch.clamp(mu_logsig[:, 0], 0.6931, 7.4593)

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
    """
class UNet3D(nn.Module):
    def __init__(self, input_shape, network_depth, no_convolutions, conv_filter_no_init, 
                 conv_kernel_size, latent_representation_dim, output_channels=1, l1=0.0, l2=0.0, 
                 dropout_value=0.0, use_batch_normalization=False, activation='relu'):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder = encoder(
            input_shape=input_shape,
            network_depth=network_depth,
            no_convolutions=no_convolutions,
            conv_filter_no_init=conv_filter_no_init,
            conv_kernel_size=conv_kernel_size,
            latent_representation_dim=latent_representation_dim,
            l1=l1,
            l2=l2,
            dropout_value=dropout_value,
            use_batch_normalization=use_batch_normalization,
            activation=activation
        )
        
        # Calculate the shape of the feature map at the end of the encoder
        conv_shape = self.encoder.feature_map_size
        print(f'this is conv shape: {conv_shape}')
        # Decoder
        self.decoder = Decoder3D(
            conv_shape=conv_shape[1:],  # Remove the batch dimension
            network_depth=network_depth,
            no_convolutions=no_convolutions,
            conv_filter_no_init=conv_filter_no_init,
            conv_kernel_size=conv_kernel_size,
            latent_representation_dim=latent_representation_dim,
            output_channels=output_channels,
            l1=l1,
            l2=l2,
            dropout_value=dropout_value,
            use_batch_normalization=use_batch_normalization,
            activation=activation
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)
        skip_connections = list(self.encoder.skip_connections) 
        for idx, i in enumerate(skip_connections):
            print(f'this is {idx} layer and its skip connection is {i.shape}')
        # Decoder forward pass with skip connections
        x = self.decoder(x, skip_connections)
        
        return x

class GlioNet(nn.Module):
    def __init__(self, Unet, latent_param_model):
        super(GlioNet, self).__init__()
        self.encoder = Unet.encoder
        self.decoder = Unet.decoder
        self.latent_param_model = latent_param_model

    def forward(self, x):
        latent_representation = self.encoder(x)
        skip_connections = list(self.encoder.skip_connections) 
        for idx, i in enumerate(skip_connections):
            print(f'this is {idx} layer and its skip connection is {i.shape}')
        reconstruction = self.decoder(latent_representation, skip_connections)
        print("this is the latent representation", latent_representation)
        latent_params = self.latent_param_model(latent_representation)
        return reconstruction, latent_params

