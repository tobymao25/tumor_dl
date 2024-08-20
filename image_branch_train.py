import torch
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from image_branch_model import encoder, Decoder3D, LatentParametersModel, GlioNet
from image_branch_model import reconstruction_loss, survival_loss


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
    survival_loss = survival_loss_fn(mu, sigma, x, delta)
    total_loss = reconstruction_loss.mean() + survival_loss.mean()
    return total_loss, reconstruction_loss.mean(), survival_loss.mean()


def train_model(config, optimizer, train_loader, compute_combined_loss, device="cpu"):
    model = model.to(device)
    model.train()
    epochs = config["epochs"]
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, survival_times) in enumerate(train_loader):
            inputs = inputs.to(device)
            survival_times = survival_times.to(device)
            delta = torch.ones_like(survival_times).to(device)
            optimizer.zero_grad()
            model,optimizer = model_fn(config)
            reconstruction, latent_params = model(inputs)
            total_loss, rec_loss, surv_loss = compute_combined_loss(
                reconstruction, inputs, latent_params, survival_times,
                reconstruction_loss, survival_loss, delta
            )
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {running_loss/len(train_loader):.4f}")

    print("Training Finished!")


def run_hyperparameter_search():
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=30, 
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        train_model, 
        config=search_space, # TODO -
        num_samples=10,  
        scheduler=scheduler,  
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0} 
    )

    print("Best hyperparameters found were: ", analysis.best_config)