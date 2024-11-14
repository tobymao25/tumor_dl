import torch
import torch.nn as nn
import torch.nn.functional as F
from image_branch_utils import GaussianNoise

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1, in_channels=5, initial_filters=64, 
                 gaussian_noise_factor=0.05, dropout_value=0.0, latent=True):
        super(ResNet3D, self).__init__()
        self.in_channels = initial_filters
        self.gaussian_noise_factor = gaussian_noise_factor
        self.latent = latent
        
        if gaussian_noise_factor:
            self.noise_layer = GaussianNoise(noise_factor=gaussian_noise_factor)
        else:
            self.noise_layer = None
        
        self.conv1 = nn.Conv3d(in_channels, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(initial_filters)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, initial_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(block, initial_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, initial_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, initial_filters * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(initial_filters * 8, num_classes)
        
        if dropout_value:
            self.dropout = nn.Dropout3d(p=dropout_value)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.noise_layer:
            x = self.noise_layer(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.layer2(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.layer3(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.layer4(out)
        if self.dropout:
            out = self.dropout(out)
        
        out = self.avgpool(out)
        latent_features = torch.flatten(out, 1)

        if self.latent:
            return latent_features
        else:
            out = F.relu(self.fc(latent_features))
            return out
        
class ClinicalCovariateModel(nn.Module):
    def __init__(self, input_dim, network_depth=3, no_units=128, dropout_value=0.3, use_batch_norm=True, latent_dim=64):
        super(ClinicalCovariateModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_value = dropout_value
        self.use_batch_norm = use_batch_norm

        for i in range(network_depth):
            self.layers.append(nn.Linear(input_dim if i == 0 else no_units, no_units))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(no_units))
        self.latent_layer = nn.Linear(no_units, latent_dim)
        self.output_layer = nn.Linear(latent_dim, 1)
        if dropout_value:
            self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        
        if self.dropout_value:
            x = self.dropout(x)

        latent_features = F.relu(self.latent_layer(x))

        survival_prediction = F.relu(self.output_layer(latent_features))
        
        return latent_features, survival_prediction

class SurvivalEnsembleModel(nn.Module):
    """this model combines imaging model and cov model, apply a neural network, to make the final survival outcome
    prediction. 
    """
    def __init__(self, resnet_model, covariate_model, ensemble_units=128, ensemble_depth=2, trainable=True):
        super(SurvivalEnsembleModel, self).__init__()

        self.resnet_model = resnet_model
        self.covariate_model = covariate_model

        for param in self.resnet_model.parameters():
            param.requires_grad = trainable
        for param in self.covariate_model.parameters():
            param.requires_grad = trainable
        
        resnet_latent_dim = resnet_model.fc.in_features  
        covariate_latent_dim = covariate_model.latent_layer.out_features 
        self.ensemble_layers = nn.ModuleList()
        input_dim = resnet_latent_dim + covariate_latent_dim  
        
        for _ in range(ensemble_depth):
            dense_layer = nn.Linear(input_dim, ensemble_units)
            torch.nn.init.xavier_uniform_(dense_layer.weight)
            self.ensemble_layers.append(dense_layer)
            input_dim = ensemble_units  
        self.output_layer = nn.Linear(ensemble_units, 1)

    def forward(self, image_data, covariate_data):
        resnet_latent_features = self.resnet_model(image_data)
        covariate_latent_features, _ = self.covariate_model(covariate_data)
        combined_features = torch.cat((resnet_latent_features, covariate_latent_features), dim=1)
        x = combined_features
        for layer in self.ensemble_layers:
            x = F.relu(layer(x))
        survival_prediction = self.output_layer(x)
        return survival_prediction


def get_resnet_layers(depth):
    if depth == 18:
        layers = [2, 2, 2, 2]  
    elif depth == 34:
        layers = [3, 4, 6, 3]  
    elif depth == 50:
        layers = [3, 4, 6, 3]  
    elif depth == 101:
        layers = [3, 4, 23, 3]  
    elif depth == 152:
        layers = [3, 8, 36, 3] 
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}. Choose from [18, 34, 50, 101, 152].")
    
    return layers

