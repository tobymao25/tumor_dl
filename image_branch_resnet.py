import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os
from image_branch_utils import GBMdataset
import torch.optim as optim
from torch.utils.data import DataLoader

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
    def __init__(self, block, layers, num_classes=1, in_channels=5, initial_filters=64):
        super(ResNet3D, self).__init__()
        self.in_channels = initial_filters
        
        self.conv1 = nn.Conv3d(in_channels, initial_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(initial_filters)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, initial_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(block, initial_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, initial_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, initial_filters * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(initial_filters * 8, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

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

def train_model(model, dataloader, criterion, optimizer, plot_output_dir, num_epochs=25, device='cuda'):
    model = model.to(device)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch+2), loss_history, label="Training Loss", marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.legend()

        # Save plot to the specified directory
        plot_filename = os.path.join(plot_output_dir, f'loss_epoch_{epoch+1}.png')
        plt.savefig(plot_filename)
        plt.close()
    print('Training complete')

def main():
    num_epochs = 25
    batch_size = 4
    learning_rate = 0.001
    depth = 50  
    
    # Define dataset and dataloader (Assume GBMdataset class is implemented)
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
    train_dataset = GBMdataset(image_dir=image_dir, csv_path=csv_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Get layers based on depth
    layers = get_resnet_layers(depth)
    
    # Instantiate model based on ResNet depth
    model = ResNet3D(ResidualBlock, layers, num_classes=1, in_channels=5, initial_filters=64)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Since it's a regression task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, plot_output_dir=loss_plot_out_dir, num_epochs=num_epochs)
