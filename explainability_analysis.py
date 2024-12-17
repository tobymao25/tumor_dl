import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

"""this is a code for explainability analysis for our image model, 
written by Yuncong Mao in November 2024"""

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    return image


def overlay_heatmap_on_image(image, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = heatmap * 0.1 + np.array(image) * 0.9
    return overlayed_image.astype(np.uint8)

###example usage###
# here I created an image with a checkerboard pattern
def create_checkerboard(size=224, square_size=28):
    pattern = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, square_size * 2):
        for j in range(0, size, square_size * 2):
            pattern[i:i+square_size, j:j+square_size] = 255  # White square
            if j + square_size < size:
                pattern[i+square_size:i+square_size*2, j+square_size:j+square_size*2] = 255  # White square
    return pattern

pattern_image = create_checkerboard()
original_image = Image.fromarray(pattern_image)  
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(original_image).unsqueeze(0)

model = models.resnet50(pretrained=True)  # replace with our resent model, here I used a pretrained resent-50 as an example
target_layer = model.layer3[-1]  # Choose the last layer as a latent feature
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
grayscale_cam = cam(input_tensor=input_tensor, targets=None)  
grayscale_cam = grayscale_cam[0]
original_image_np = np.array(original_image) / 255.0
cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

## Plot the results ##
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image_np)
axes[0].set_title("Patterned Image")
axes[0].axis('off')

# heatmap
axes[1].imshow(grayscale_cam, cmap='jet')
axes[1].set_title("Heatmap")
axes[1].axis('off')

# Plot the heatmap overlaid on the pattern image
axes[2].imshow(cam_image)
axes[2].set_title("Heatmap Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.show()
