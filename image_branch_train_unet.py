import os
import pandas as pd
import torch
import torch.optim as optim
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from image_branch_utils import GBMdataset
from image_branch_unet import LatentParametersModel, GlioNet, UNet3D
from image_branch_unet import reconstruction_loss, survival_loss
#import torchvision.transforms as T


def model_fn(config):
    unet_model = UNet3D(
        input_shape=config["input_shape"],
        network_depth=config["network_depth"],
        no_convolutions=config["no_convolutions"],
        conv_filter_no_init=config["conv_filter_no_init"],
        conv_kernel_size=config["conv_kernel_size"],
        latent_representation_dim=config["latent_representation_dim"],
        output_channels=1, # new
        dropout_value=config["dropout_value"],
        use_batch_normalization=config["use_batch_normalization"],
        activation=config["activation"])

    latent_param_model = LatentParametersModel(
        latent_representation_dim=config["latent_representation_dim"])

    model = GlioNet(unet_model, latent_param_model)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.01) # weight decay added

    return model, optimizer


def compute_combined_loss(reconstruction, target, latent_params, x, 
                          reconstruction_loss_fn, survival_loss_fn):
    mu = latent_params
    #plot_survival_curve(mu.detach().cpu()[0])
    reconstruction_loss = reconstruction_loss_fn(target, reconstruction)
    MSE = survival_loss_fn(mu, x)
    total_loss = reconstruction_loss.mean()+MSE.mean()
    return total_loss, reconstruction_loss.mean(), MSE.mean()


def plot_loss_curves(loss_plot_out_dir, train_epoch_losses, valid_epoch_losses):

    # Report and print epoch losses
    for key in train_epoch_losses:  
        plt.figure(figsize=(5, 3))
  
        train_losses = train_epoch_losses[key]
        train_losses = train_losses[1:] # do not plot the loss of the first epoch
        plt.plot([x+1 for x in range(len(train_losses))], train_losses, label="train_"+key, color="blue", lw=3)

        valid_losses = valid_epoch_losses[key]
        valid_losses = valid_losses[1:] # do not plot the loss of the first epoch
        plt.plot([x+1 for x in range(len(valid_losses))], valid_losses, label="valid_"+key, color="red", lw=3)
        
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        # Save images
        plt.savefig(os.path.join(loss_plot_out_dir, f"{key} curve {timestamp}.png"))


def train_model(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_w_hopkins.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
    train_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_train.csv"
    valid_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_test.csv"
    # transform = T.Compose([
    # T.ToTensor(),  # Convert to tensor
    # T.Normalize(mean=[0.0], std=[1.0]) ])

    # train valid split
    all_patient_data_df = pd.read_csv(csv_path)
    random_seed = 11
    print("the random seed is 11")
    train_df, valid_df = train_test_split(all_patient_data_df, test_size=0.1, random_state=random_seed)
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)

    # setup training and validation datasets
    train_dataset = GBMdataset(image_dir=image_dir, csv_path=train_csv_path)#, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_dataset = GBMdataset(image_dir=image_dir, csv_path=valid_csv_path)#, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=4)

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # set up model and optimizer
    model, optimizer = model_fn(config)
    model = model.to(device)

    # If multiple GPUs are available, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print("Using a single GPU")

    epochs = config["epochs"]
    epoch_losses = {"loss": [], "MSE": [], "rec_loss": []}
    epoch_validation_losses = {"loss": [], "MSE": [], "rec_loss": []}

    for epoch in range(epochs):
        print(f"Current epoch: {epoch+1}")
        running_losses = {"loss": 0.0, "MSE": 0.0, "rec_loss": 0.0}
        model.train()
        for i, (inputs, survival_times) in enumerate(train_dataloader):
            print("batch", i, "out of", len(train_dataloader))
            # print("Model parameters:")
            # for param in model.parameters():
            #     print(param)
            # process inputs data
            inputs = inputs.to(device)
            inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
            survival_times = survival_times.to(device)
            delta = torch.ones_like(survival_times).to(device)
            
            reconstruction, latent_params = model(inputs)

            # Save tensors and reconstructions every 30 epochs
            if epoch % 30 == 0 and i == 0:
                torch.save(inputs,  f'/home/ltang35/tumor_dl/TrainingDataset/out/inputs_tensor_epoch{epoch}.pt')
                torch.save(reconstruction, f'/home/ltang35/tumor_dl/TrainingDataset/out/reconstruction_tensor_epoch{epoch}.pt')
            
            total_loss, rec_loss, MSE = compute_combined_loss(
                reconstruction, inputs, latent_params, survival_times,
                reconstruction_loss, survival_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_losses["loss"] += total_loss.item()
            running_losses["MSE"] += MSE.item()
            running_losses["rec_loss"] += rec_loss.item()

        # Report and print epoch losses
        for key in running_losses:
            avg_loss = running_losses[key] / len(train_dataloader)
            # if key == "loss":
            #     train.report({key: avg_loss})
            print(f"Epoch {epoch+1}/{epochs}, {key}: {avg_loss:.4f}")
            epoch_losses[key].append(avg_loss)

        # Save the model checkpoint every 30 epochs
        if epoch % 30 == 0:
            torch.save(model.state_dict(), f'/home/ltang35/tumor_dl/TrainingDataset/out/model_epoch_{epoch}.pt')

        print(f"--Validation for epoch {epoch+1}--")
        model.eval()
        with torch.no_grad():
            running_losses = {"loss": 0.0, "MSE": 0.0, "rec_loss": 0.0}
            for i, (inputs, survival_times) in enumerate(valid_dataloader):
                print("batch", i, "out of", len(valid_dataloader))
                # process inputs data
                inputs = inputs.to(device)
                inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
                survival_times = survival_times.to(device)
                delta = torch.ones_like(survival_times).to(device)
            
                pred_reconstruction, pred_latent_params = model(inputs)
                valid_total_loss, valid_rec_loss, valid_MSE = compute_combined_loss(
                    pred_reconstruction, inputs, pred_latent_params, survival_times,
                    reconstruction_loss, survival_loss
                )

                running_losses["loss"] += valid_total_loss.item()
                running_losses["MSE"] += valid_MSE.item()
                running_losses["rec_loss"] += valid_rec_loss.item()

            # Report and print epoch losses
            for key in running_losses:
                avg_loss = running_losses[key] / len(valid_dataloader)
                print(f"Epoch {epoch+1}/{epochs}, {key}: {avg_loss:.4f}")
                epoch_validation_losses[key].append(avg_loss)
        print(f"--Validation finished for epoch {epoch+1}--")
    
        plot_loss_curves(loss_plot_out_dir, epoch_losses, epoch_validation_losses) # plot loss curves after each epoch

        # Clear GPU cache
        torch.cuda.empty_cache()

    print("Training Finished!")


def run_hyperparameter_search(search_space, num_samples):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=30, 
        grace_period=5,
        reduction_factor=2
    )

    analysis = tune.run(
        train_model, 
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,  
        verbose=1, 
        resources_per_trial={"cpu": 10, "gpu": 1 if torch.cuda.is_available() else 0}, 
        storage_path="/home/ltang35/tumor_dl/output", 
    )

    best_trial = analysis.get_best_trial(metric="loss", mode="min", scope="last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")


