import logging
import ray
import ray.tune as tune
#from image_branch_train_unet import train_model, run_hyperparameter_search
from image_branch_train_resnet import train_resnet

if __name__ =='__main__':
    
    ##unet
    # config = {
    # 'input_shape': (4, 128, 128, 128), 
    # 'network_depth': 4, # 2, 4, 6, 8
    # 'no_convolutions': 3, #3-6
    # 'conv_filter_no_init': 12, 
    # 'conv_kernel_size': 3, 
    # 'latent_representation_dim': 256, #512
    # 'dropout_value': 0.1, 
    # 'use_batch_normalization': True, 
    # 'activation': "leakyrelu", 
    # "lr": 1e-5, 
    # "epochs": 3000
    # }
    # print(config)
    # train_model(config=config)

    config = {
    'batch_size': 4, 
    'depth': 50, 
    "lr": 1e-5, 
    "epochs": 3000
    }
    print(config)
    train_resnet(config=config)

    # --- for hyperparameter tuning
    # search_space = {
    # "input_shape": (5, 128, 128, 128),  # Fixed
    # "network_depth": tune.choice([2, 4]), # Only supports 2 or 4
    # "no_convolutions": tune.choice([2, 6]), #, 3
    # "conv_filter_no_init": tune.choice([32, 64]),
    # "conv_kernel_size": tune.choice([3, 5]), #, 5
    # "latent_representation_dim": tune.choice([128, 256]), 
    # "dropout_value": tune.uniform(0.1, 0.4), #0.4
    # "use_batch_normalization": tune.choice([True, False]),
    # "activation": tune.choice(["relu", "leakyrelu"]),
    # "lr": tune.uniform(1e-5, 1e-4), 
    # "epochs": 50 #tune.randint(30, 100),  # likely > 50, previous paper used 2000
    # }

    # ray.init()
    # logging.getLogger("ray").setLevel(logging.INFO)
    # run_hyperparameter_search(search_space=search_space, num_samples=10)

    
    ###################### For plotting  and debugging labels
    # import torchio as tio
    # import numpy as np
    # import matplotlib.pyplot as plt

    # seg = tio.ScalarImage("/home/ltang35/tumor_dl/TrainingDataset/labels/Brats17_TCIA_175_1_seg.nii")
    # seg = seg.data
    # print(seg.shape, "seg shape")
    # # Get the unique values and their counts
    # unique_values, counts = np.unique(seg, return_counts=True)
    # # Display the results
    # for value, count in zip(unique_values, counts):
    #     print(f"Value {value} appears {count} times")
        
    # # Select a slice (e.g., from the middle of the volume)
    # slice_index = seg.shape[1] // 2  # Middle slice along the depth axis
    # seg = seg[0, slice_index-15, :, :]  # Selecting the first channel and middle slice

    # # Plot the selected slice
    # plt.imshow(seg, cmap='viridis', interpolation='none')
    # plt.colorbar(label='Integer Value')
    # plt.savefig("/home/ltang35/tumor_dl/integers_seg.png")


    # t1ce = tio.ScalarImage("/home/ltang35/tumor_dl/TrainingDataset/images/Brats17_TCIA_175_1_t1ce.nii")
    # t1ce = t1ce.data
    # print(t1ce.shape, "t1ce shape")
    # # Select a slice (e.g., from the middle of the volume)
    # slice_index = t1ce.shape[1] // 2  # Middle slice along the depth axis
    # t1ce = t1ce[0, slice_index-15, :, :]  # Selecting the first channel and middle slice

    # # Plot the selected slice
    # plt.imshow(t1ce, cmap="gray")
    # plt.savefig("/home/ltang35/tumor_dl/integers_seg_t1ce_correspond.png")
    

    #--- for plotting reconstruction
    # import torch
    # import matplotlib.pyplot as plt
    # input1 =  torch.load('/home/ltang35/tumor_dl/TrainingDataset/out/inputs_tensor_epoch1270.pt')
    # recon = torch.load('/home/ltang35/tumor_dl/TrainingDataset/out/reconstruction_tensor_epoch1270.pt')

    # # Step 2: Convert tensor to NumPy array
    # input1 = input1.cpu().detach().numpy()
    # print(input1.shape)
    # input1 = input1[0, 0, :, :, 75]

    # # Step 2: Convert tensor to NumPy array
    # recon = recon.cpu().detach().numpy()
    # print(recon.shape)
    # recon = recon[0, 0, :, :, 75]

    # # Step 3: Plot the data using matplotlib
    # plt.figure(figsize=(10, 6))
    # plt.imshow(input1, cmap="gray")
    # plt.savefig("/home/ltang35/tumor_dl/plot_input.png")

    # plt.figure(figsize=(10, 6))
    # plt.imshow(recon, cmap="gray")
    # plt.savefig("/home/ltang35/tumor_dl/plot_recon.png")

    # --- for debugging why there are 3s in the labels. 
    # import os
    # import torchio as tio
    # import numpy as np

    # # Iterate through all files in the folder
    # for file in os.listdir("/home/ltang35/tumor_dl/TrainingDataset/labels/"):
    #     if file.endswith('.nii') or file.endswith('.nii.gz'):  # Check if the file is a NIfTI file
    #         file_path = os.path.join("/home/ltang35/tumor_dl/TrainingDataset/labels/", file)
            
    #         # Load the NIfTI file
    #         nii_img = tio.ScalarImage(file_path)

    #         print(nii_img.shape, "nii_data shape_og")
    #         print(file)

    #         resample_transform = tio.transforms.Resample((1, 1, 1), image_interpolation='nearest')
    #         nii_img = resample_transform(nii_img)
    #         resize_transform = tio.transforms.Resize((128, 128, 128), image_interpolation='nearest')
    #         nii_img = resize_transform(nii_img)
    #         nii_img = nii_img.data.numpy()

    #         print(nii_img.shape, "nii_data shape")
    #         print(file)

    #         # Get the unique values and their counts
    #         unique_values, counts = np.unique(nii_img, return_counts=True)
    #         # Display the results
    #         for value, count in zip(unique_values, counts):
    #             print(f"Value {value} appears {count} times")
    