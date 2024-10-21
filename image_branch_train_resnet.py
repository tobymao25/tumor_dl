import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from image_branch_utils import GBMdataset
from image_branch_resnet import ResNet3D, ResidualBlock, get_resnet_layers

def plot_loss_curves(loss_plot_out_dir, train_epoch_losses):

    # Report and print epoch losses 
    plt.figure(figsize=(5, 3))

    train_losses = train_epoch_losses
    train_losses = train_losses[1:] # do not plot the loss of the first epoch
    plt.plot([x+1 for x in range(len(train_losses))], train_losses, label="train", color="blue", lw=3)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    # Save images
    plt.savefig(os.path.join(loss_plot_out_dir, f"Loss {timestamp}.png"))


def train_resnet(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"

    # setup training and validation datasets
    dataset = GBMdataset(image_dir=image_dir, csv_path=csv_path)#, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
   
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # set up model and optimizer
    layers = get_resnet_layers(config['depth'])
    model = ResNet3D(ResidualBlock, layers, num_classes=1, in_channels=5, initial_filters=64)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss(reduction='mean')

    # If multiple GPUs are available, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print("Using a single GPU")

    epochs = config["epochs"]
    epoch_losses = []

    for epoch in range(epochs):
        print(f"Current epoch: {epoch+1}")
        running_losses = 0.0
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
            
            outputs = model(inputs)
            print("here are the predictions", outputs)
            print("here are the labels", survival_times)
            loss = criterion(outputs, survival_times)
            print("This is the calculated loss", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_losses += loss.item() * inputs.size(0) # in case the last batch has fewer samples

        # Report and print epoch losses
        avg_loss = running_losses / len(dataloader.dataset)
        # if key == "loss":
        #     train.report({key: avg_loss})
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

        plot_loss_curves(loss_plot_out_dir, epoch_losses) # plot loss curves after each epoch

        # Clear GPU cache
        torch.cuda.empty_cache()

    print("Training Finished!")
