import os
import torch
import torch.optim as optim
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from image_branch_utils import GBMdataset
from image_branch_model import encoder, Decoder3D, LatentParametersModel, GlioNet, UNet3D
from image_branch_model import reconstruction_loss, survival_loss
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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    return model, optimizer


def compute_combined_loss(reconstruction, target, latent_params, x, 
                          reconstruction_loss_fn, survival_loss_fn):
    mu = latent_params
    #plot_survival_curve(mu.detach().cpu()[0])
    reconstruction_loss = reconstruction_loss_fn(target, reconstruction)
    MSE = survival_loss_fn(mu, x)
    total_loss = reconstruction_loss.mean()+MSE.mean()
    return total_loss, reconstruction_loss.mean(), MSE.mean()


def plot_loss_curves(loss_plot_out_dir, epoch_losses):

    # Report and print epoch losses
    for key in epoch_losses:  
        plt.figure(figsize=(5, 3))

        losses = epoch_losses[key]
        losses = losses[1:] # do not plot the loss of the first epoch
        plt.plot([x+1 for x in range(len(losses))], losses, label=key, lw=3)
        plt.xlabel('Epochs')
        plt.ylabel(key)

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        plt.savefig(os.path.join(loss_plot_out_dir, f"{key} curve {timestamp}.png"))

"""
def plot_survival_curve(mu, sigma):
    t = torch.arange(0,1731)
    S = S = 1 / (1 + torch.exp((torch.log(t) - mu) / sigma))
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label=f'Survival Probability (mu={mu.item()}, sigma={sigma.item()})')
    plt.xlabel('t (Survival Time)')
    plt.ylabel('S (Survival Probability)')
    plt.title('Survival Probability vs. Survival Time')
    plt.grid(True)
    plt.legend()
    plt.savefig("/home/ltang35/tumor_dl/TrainingDataset/out/survival_curve.png")
"""


def train_model(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
    # transform = T.Compose([
    # T.ToTensor(),  # Convert to tensor
    # T.Normalize(mean=[0.0], std=[1.0]) ])

    dataset = GBMdataset(image_dir=image_dir, csv_path=csv_path)#, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

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

    epoch_losses = {"loss": [], "MSE": [], "rec_loss": []}
    epochs = config["epochs"]

    for epoch in range(epochs):
        print(f"Current epoch: {epoch+1}")
        running_losses = {"loss": 0.0, "MSE": 0.0, "rec_loss": 0.0}
        model.train()
        for i, (inputs, survival_times) in enumerate(dataloader):
            print("batch", i, "out of", len(dataloader))
            # print("Model parameters:")
            # for param in model.parameters():
            #     print(param)
            # process inputs data
            inputs = inputs.to(device)
            inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
            survival_times = survival_times.to(device)
            delta = torch.ones_like(survival_times).to(device)
            
            reconstruction, latent_params = model(inputs)

            # Save tensors and reconstructions every 10 epochs
            if epoch % 10 == 0 and i == 0:
                torch.save(inputs,  f'/home/ltang35/tumor_dl/TrainingDataset/out/inputs_tensor_epoch{epoch}.pt')
                torch.save(reconstruction, f'/home/ltang35/tumor_dl/TrainingDataset/out/reconstruction_tensor_epoch{epoch}.pt')
            
            total_loss, rec_loss, MSE = compute_combined_loss(
                reconstruction, inputs, latent_params, survival_times,
                reconstruction_loss, survival_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            # for name, param in model.named_parameters():
            #     print(f"\n{name} - Shape: {param.shape}")
            #     print("Weights:")
            #     print(param.data)  # Weights or parameters
            #     if param.grad is not None:
            #         print("Gradients:")
            #         print(param.grad)  # Gradients
            #     else:
            #         print("No gradients for this parameter.")

            optimizer.step()

            running_losses["loss"] += total_loss.item()
            running_losses["MSE"] += MSE.item()
            running_losses["rec_loss"] += rec_loss.item()

        # Report and print epoch losses
        for key in running_losses:
            avg_loss = running_losses[key] / len(dataloader)
            if key == "loss":
                train.report({key: avg_loss})
            print(f"Epoch {epoch+1}/{epochs}, {key}: {avg_loss:.4f}")
            epoch_losses[key].append(avg_loss)

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Save the model checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'/home/ltang35/tumor_dl/TrainingDataset/out/model_epoch_{epoch}.pt')

        # Plot loss curves after each epoch
        plot_loss_curves(loss_plot_out_dir, epoch_losses)

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

