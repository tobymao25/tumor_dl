import logging
import ray
import ray.tune as tune
from image_branch_train import train_model, run_hyperparameter_search

if __name__ =='__main__':
    
    config = {
    'input_shape': (5, 128, 128, 128), 
    'network_depth': 4, # 2, 4, 6, 8
    'no_convolutions': 3, #3-6
    'conv_filter_no_init': 12, 
    'conv_kernel_size': 3, 
    'latent_representation_dim': 256, #512
    'dropout_value': 0.1, 
    'use_batch_normalization': True, 
    'activation': "leakyrelu", 
    "lr": 1e-5, 
    "epochs": 300
    }

    print(config)
    train_model(config=config)


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


    #--- for plotting reconstruction
    # import torch
    # import matplotlib.pyplot as plt
    # input1 =  torch.load('/home/ltang35/tumor_dl/TrainingDataset/out/inputs_tensor_epoch40.pt')
    # recon = torch.load('/home/ltang35/tumor_dl/TrainingDataset/out/reconstruction_tensor_epoch40.pt')

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
