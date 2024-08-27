import os
import torch
import torch.optim as optim
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from image_branch_utils import GBMdataset
from image_branch_model import encoder, Decoder3D, LatentParametersModel, GlioNet
from image_branch_model import reconstruction_loss, survival_loss
import torchvision.transforms as T

def compile_all_models(input_shape,
                        network_depth,
                        no_convolutions,
                        conv_filter_no_init,
                        conv_kernel_size,
                        latent_representation_dim,
                        dropout_value,
                        use_batch_normalization,
                        activation,
                        l1,
                        l2,
                        lr=1e-3):
    
    encoder_model = encoder(
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
        activation=activation,
        gaussian_noise_factor = 0.3
    )

    conv_shape = encoder_model.feature_map_size[1:]
    decoder_model = Decoder3D(
        conv_shape=conv_shape,
        network_depth=network_depth,
        no_convolutions=no_convolutions,
        conv_filter_no_init=conv_filter_no_init,
        conv_kernel_size=conv_kernel_size,
        latent_representation_dim=latent_representation_dim,
        dropout_value=dropout_value,
        use_batch_normalization=use_batch_normalization,
        activation=activation
    )

    latent_param_model = LatentParametersModel(
        latent_representation_dim=latent_representation_dim,
        l1=l1,
        l2=l2
    )

    model = GlioNet(encoder_model, decoder_model, latent_param_model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def model_fn(config):
    model, optimizer = compile_all_models(
        input_shape=config["input_shape"],
        network_depth=config["network_depth"],
        no_convolutions=config["no_convolutions"],
        conv_filter_no_init=config["conv_filter_no_init"],
        conv_kernel_size=config["conv_kernel_size"],
        latent_representation_dim=config["latent_representation_dim"],
        dropout_value=config["dropout_value"],
        use_batch_normalization=config["use_batch_normalization"],
        activation=config["activation"],
        l1=0.0,
        l2=0.0,
        lr=config["lr"]
    )
    return model, optimizer


def compute_combined_loss(reconstruction, target, latent_params, x, 
                          reconstruction_loss_fn, survival_loss_fn, delta=1):
    
    mu, sigma = latent_params[:, 0], latent_params[:, 1]
    reconstruction_loss = reconstruction_loss_fn(target, reconstruction)
    survival_loss, MSE = survival_loss_fn(mu, sigma, x, delta)
    total_loss = reconstruction_loss.mean() + survival_loss.mean()
    return total_loss, reconstruction_loss.mean(), survival_loss.mean(), MSE


def plot_loss_curves(loss_plot_out_dir, epoch_losses):

    # Report and print epoch losses
    for key in epoch_losses:  
        plt.figure(figsize=(5, 3))

        losses = [loss.cpu().item() for loss in epoch_losses[key]]
        plt.plot([x+1 for x in range(len(losses))], losses, label=key, lw=3)
        plt.xlabel('Epochs')
        plt.ylabel(key)

        plt.savefig(os.path.join(loss_plot_out_dir, f"{key} curve.png"))


def train_model(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset"
    transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
    T.Normalize(mean=[0.0], std=[1.0]) ])

    dataset = GBMdataset(image_dir=image_dir, csv_path=csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

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

    epoch_losses = {"loss": [], "MSE": [], "rec_loss": [], "surv_loss": []}
    epochs = config["epochs"]
    for epoch in range(epochs):
        print("current epoch:", epoch+1)
        running_losses = {"loss": 0.0, "MSE": 0.0, "rec_loss": 0.0, "surv_loss": 0.0}
        model.train()
        for i, (inputs, survival_times) in enumerate(dataloader):
            # process inputs data
            inputs = inputs.to(device)
            #inputs = inputs.squeeze(2) #no need since changed the dataloader
            survival_times = survival_times.to(device)
            delta = torch.ones_like(survival_times).to(device)
            
            optimizer.zero_grad()
            reconstruction, latent_params = model(inputs)
            total_loss, rec_loss, surv_loss, MSE = compute_combined_loss(
                reconstruction, inputs, latent_params, survival_times,
                reconstruction_loss, survival_loss, delta
            )
            total_loss.backward()
            optimizer.step()
            running_losses["loss"] += total_loss.item()
            running_losses["MSE"] += MSE
            running_losses["rec_loss"] += rec_loss
            running_losses["surv_loss"] += surv_loss

        # Report and print epoch losses
        for key in running_losses:
            avg_loss = running_losses[key] / len(dataloader)
            if key == "loss":
                train.report({key: avg_loss})
            print(f"Epoch {epoch+1}/{epochs}, {key}: {avg_loss:.4f}")
            epoch_losses[key].append(running_losses[key])

        torch.cuda.empty_cache() 
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

