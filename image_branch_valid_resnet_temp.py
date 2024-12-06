import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from image_branch_utils import GBMdataset
from image_branch_resnet import ResNet3D, ResidualBlock, get_resnet_layers

# def plot_valid_loss_curves(loss_plot_out_dir, epoch_validated, epoch_valid_losses):

#     plt.figure(figsize=(5, 3))
#     plt.plot(epoch_validated, epoch_valid_losses, label="valid", color="red", lw=2)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Get the current timestamp
#     timestamp = datetime.now().strftime('%Y%m%d')
#     # Save images
#     plt.savefig(os.path.join(loss_plot_out_dir, f"Loss_Valid_retro_using_saved_model_{timestamp}.png"))
#     plt.close()

# def valid_resnet(): 

#     # setup data
#     image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
#     loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
#     valid_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_test.csv"

#     # setup training and validation datasets
#     valid_dataset = GBMdataset(image_dir=image_dir, csv_path=valid_csv_path)#, transform=transform)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=4)
   
#     # setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     model_paths = [
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_0.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_30.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_60.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_90.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_120.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_150.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_180.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_210.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_240.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_270.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_300.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_330.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_360.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_390.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_420.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_450.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_480.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_510.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_540.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_570.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_600.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_630.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_660.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_690.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_720.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_750.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_780.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_810.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_840.pt", 
#         "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_870.pt"
#     ]
    
#     epoch_validated = [ 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 
#                        480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870]
                            
#     epoch_valid_losses = []
#     for i in model_paths:

#         # set up model and optimizer
#         layers = get_resnet_layers(50) ## hard code
#         model = ResNet3D(ResidualBlock, layers, num_classes=1, in_channels=5, initial_filters=64)
#         state_dict = torch.load(i)
#         model.load_state_dict(state_dict)
#         model = model.to(device)
#         model.eval()
#         criterion = nn.MSELoss(reduction='mean')

#         # If multiple GPUs are available, wrap the model with DataParallel
#         if torch.cuda.device_count() > 1:
#             print(f"Using {torch.cuda.device_count()} GPUs")
#             model = torch.nn.DataParallel(model)
#         else:
#             print("Using a single GPU")

#         with torch.no_grad():
#             running_losses = 0.0
#             for i, (inputs, survival_times) in enumerate(valid_dataloader):
#                 print("batch", i, "out of", len(valid_dataloader))
#                 # print("Model parameters:")
#                 # for param in model.parameters():
#                 #     print(param)
#                 # process inputs data
#                 inputs = inputs.to(device)
#                 inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
#                 survival_times = survival_times.to(device)
                
#                 outputs = model(inputs)
#                 outputs = outputs.squeeze() # Ensure mu and x has same dimension
#                 print("here are the predictions", outputs)
#                 print("here are the labels", survival_times)
#                 loss = criterion(outputs, survival_times)
#                 print("This is the calculated loss", loss)

#                 running_losses += loss.item() * inputs.size(0) # in case the last batch has fewer samples

#             # Report and print epoch losses
#             avg_loss = running_losses / len(valid_dataloader.dataset)
#             print("loss: {avg_loss:.4f}")
#             epoch_valid_losses.append(avg_loss)

#             print("model_validated", i)
#             print("epoch_valid_losses", epoch_valid_losses)

#     plot_valid_loss_curves(loss_plot_out_dir, epoch_validated, epoch_valid_losses) # plot loss curves after each epoch
#     print("validation finished!!")

##### Explainability --------- --------- --------- --------- --------- --------- --------- ---------
##### Explainability --------- --------- --------- --------- --------- --------- --------- ---------
##### Explainability --------- --------- --------- --------- --------- --------- --------- ---------
##### Explainability --------- --------- --------- --------- --------- --------- --------- ---------

import shap
import numpy as np

def valid_resnet(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
    valid_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_test.csv"

    # setup training and validation datasets
    valid_dataset = GBMdataset(image_dir=image_dir, csv_path=valid_csv_path)#, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4)
   
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_paths = [
        "/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_30_exp11.pt"
    ]
    
    epoch_validated = [30]
                            
    epoch_valid_losses = []
    for i in model_paths:

        # set up model and optimizer
        layers = get_resnet_layers(18) ## hard code
        model = ResNet3D(ResidualBlock, layers, num_classes=1, in_channels=3, initial_filters=64, 
                        gaussian_noise_factor=config["noise_factor"], dropout_value=config['dropout_value'])
        state_dict = torch.load(i, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # If multiple GPUs are available, wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        else:
            print("Using a single GPU")

        with torch.no_grad():
            for i, (inputs, survival_times) in enumerate(valid_dataloader):
                print("batch", i, "out of", len(valid_dataloader))
                # print("Model parameters:")
                # for param in model.parameters():
                #     print(param)
                # process inputs data
                inputs = inputs.to(device)
                inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
                survival_times = survival_times.to(device)

                print("shape inputs", inputs.shape) #shape inputs torch.Size([8, 3, 128, 128, 128])
                print("shape inputs[0, 0, :, :, 75]", inputs[0, 0, :, :, 75].shape)


                # Step 1: Define a wrapper function to convert NumPy to PyTorch tensor and vice versa
                def model_wrapper(X):
                    # X is a NumPy array, so we convert it to a PyTorch tensor
                    X_tensor = torch.from_numpy(X).float()
                    
                    # Make predictions using the model (assuming the model is on CPU or move to GPU if needed)
                    with torch.no_grad():  # No need to compute gradients for inference
                        outputs = model(X_tensor)
                        
                    # Convert the output to a NumPy array
                    return outputs.detach().cpu().numpy()

                #outputs = model(inputs)
                #print("outputs before squeezing", outputs)
                #outputs = outputs.squeeze() # Ensure mu and x has same dimension
                #print("outputs after squeezing", outputs)

                # run explainabiilty analysis on an examples 
                background = np.random.rand(1, 3, 128, 128, 128)
                #background_tensor = torch.from_numpy(background).float() 
                explainer = shap.KernelExplainer(model_wrapper, background)
                shap_values = explainer.shap_values(inputs.numpy())

                # Convert the SHAP values to a NumPy array and visualize them
                shap_values_np = shap_values[0].detach().numpy()

                print("shape shap_values_np", shap_values_np.shape)
                print("shape shap_values_np[0, 1, :, :, 50]]", shap_values_np[0, 1, :, :, 50].shape)

                shap.image_plot([shap_values_np[0, 1, :, :, 50]], inputs.numpy()[0, 1, :, :, 50].transpose(0, 2, 3, 1))
                plt.savefig("shap_image_plot.png", dpi=300, bbox_inches='tight')
                plt.close()  # Close the plot to free up memory

                break

        

