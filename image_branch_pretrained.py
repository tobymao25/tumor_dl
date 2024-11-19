"""this code is a model using pretrained 2d resnet over 3d volume. The input of the model should be image size of 
(batch_size, 3, 128, 128) which contains t1ce, flair, and t2"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet2DTo3D(nn.Module):
    """
    Transfer learnable parameters from a pretrained 2D ResNet-18 on ImageNet
    to a 3D ResNet-18 by duplicating the 2D filters along the third dimension.
    """
    def __init__(self, in_channels=3, num_classes=1, pretrained=True, freeze_lower_layers=True):
        super(ResNet2DTo3D, self).__init__()
        resnet2d = resnet18(pretrained=pretrained)
        # Convert the first convolutional layer to 3D
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=resnet2d.conv1.out_channels,
            kernel_size=(7, 7, 7),  
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        
        # Duplicate 2D filters along the third dimension to initialize 3D filters
        self._inflate_weights(resnet2d.conv1.weight, self.conv1.kernel_size[0])

        # Convert the first batch normalization layer to BatchNorm3d
        self.bn1 = nn.BatchNorm3d(resnet2d.bn1.num_features)

        # Copy the remaining layers directly from the 2D ResNet
        self.relu = resnet2d.relu
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Convert ResNet blocks to 3D
        self.layer1 = self._convert_to_3d(resnet2d.layer1)
        self.layer2 = self._convert_to_3d(resnet2d.layer2)
        self.layer3 = self._convert_to_3d(resnet2d.layer3)
        self.layer4 = self._convert_to_3d(resnet2d.layer4)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(resnet2d.fc.in_features, num_classes)

        # Freeze lower layers if specified
        if freeze_lower_layers:
            self._freeze_layers()

    def _inflate_weights(self, conv2d_weights, depth):
        """
        Inflate 2D convolutional weights into 3D by duplicating along the depth dimension.
        """
        with torch.no_grad():
            # Add a new dimension and duplicate the weights along the depth dimension
            conv3d_weights = conv2d_weights.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # Match depth

            # Handle cases where `in_channels` is different from 3, in case wanting to add segmentations
            if self.conv1.in_channels != 3:
                conv3d_weights = conv3d_weights.repeat(1, self.conv1.in_channels // 3, 1, 1, 1)
                conv3d_weights = conv3d_weights[:, :self.conv1.in_channels, :, :, :]  # Trim extra channels

            self.conv1.weight.copy_(conv3d_weights)

    def _convert_to_3d(self, module):
        """
        Recursively convert a ResNet block (or layer) from 2D to 3D.
        Replace Conv2d and BatchNorm2d with Conv3d and BatchNorm3d.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Replace Conv2d with Conv3d, add depth to all components
                new_conv = nn.Conv3d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size + (1,),  
                    stride=child.stride + (1,),            
                    padding=child.padding + (0,),          
                    dilation=child.dilation + (1,),        
                    bias=child.bias
                )
                setattr(module, name, new_conv)
            elif isinstance(child, nn.BatchNorm2d):
                new_bn = nn.BatchNorm3d(child.num_features)
                setattr(module, name, new_bn)
            elif isinstance(child, nn.MaxPool2d):
                new_pool = nn.MaxPool3d(
                    kernel_size=child.kernel_size + (1,),
                    stride=child.stride + (1,),
                    padding=child.padding + (0,)
                )
                setattr(module, name, new_pool)
            else:
                self._convert_to_3d(child)
        return module

    def _freeze_layers(self):
        """
        Freeze the lower layers of the network (conv1, bn1, layer1, and layer2).
        """
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    