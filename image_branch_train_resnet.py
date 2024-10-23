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

def plot_loss_curves(loss_plot_out_dir, train_epoch_losses):

    # Report and print epoch losses 
    plt.figure(figsize=(5, 3))

    train_losses = train_epoch_losses
    train_losses = train_losses[1:] # do not plot the loss of the first epoch
    plt.plot([x+1 for x in range(len(train_losses))], train_losses, label="train", color="blue", lw=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    # Save images
    plt.savefig(os.path.join(loss_plot_out_dir, f"Loss_{timestamp}.png"))
    plt.close()


def plot_valid_loss_curves(loss_plot_out_dir, epoch_validated, epoch_valid_losses):

    plt.figure(figsize=(5, 3))
    plt.plot(epoch_validated, epoch_valid_losses, label="valid", color="red", lw=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    # Save images
    plt.savefig(os.path.join(loss_plot_out_dir, f"Loss_Valid_{timestamp}.png"))
    plt.close()

def train_resnet(config): 

    # setup data
    image_dir = "/home/ltang35/tumor_dl/TrainingDataset/images"
    csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin.csv"
    loss_plot_out_dir = "/home/ltang35/tumor_dl/TrainingDataset/out"
    train_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_train.csv"
    valid_csv_path = "/home/ltang35/tumor_dl/TrainingDataset/survival_data_fin_test.csv"

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
    epoch_validated = []
    epoch_valid_losses = []

    for epoch in range(epochs):
        print(f"Current epoch: {epoch+1}")
        running_losses = 0.0
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
            
            outputs = model(inputs)
            outputs = outputs.squeeze() # Ensure mu and x has same dimension
            print("here are the predictions", outputs)
            print("here are the labels", survival_times)
            loss = criterion(outputs, survival_times)
            print("This is the calculated loss", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_losses += loss.item() * inputs.size(0) # in case the last batch has fewer samples

        # Report and print epoch losses
        avg_loss = running_losses / len(train_dataloader.dataset)
        # if key == "loss":
        #     train.report({key: avg_loss})
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

        # Save the model checkpoint every 30 epochs
        if epoch % 30 == 0:
            torch.save(model.state_dict(), f'/home/ltang35/tumor_dl/TrainingDataset/out/resnet_epoch_{epoch}.pt')

        # plot loss curve for training
        plot_loss_curves(loss_plot_out_dir, epoch_losses) # plot loss curves after each epoch

        # Validate every 5 epoches
        if epoch % 5 == 0:
            print(f"--Validation for epoch {epoch+1}--")
            model.eval()
            with torch.no_grad():
                running_losses = 0.0
                #model.train()
                for i, (inputs, survival_times) in enumerate(valid_dataloader):
                    print("batch", i, "out of", len(valid_dataloader))
                    # print("Model parameters:")
                    # for param in model.parameters():
                    #     print(param)
                    # process inputs data
                    inputs = inputs.to(device)
                    inputs = inputs.squeeze(2) # remove 3rd dimension [n, 5, 1, 128, 128, 128]
                    survival_times = survival_times.to(device)
                    
                    outputs = model(inputs)
                    outputs = outputs.squeeze() # Ensure mu and x has same dimension
                    print("here are the predictions", outputs)
                    print("here are the labels", survival_times)
                    loss = criterion(outputs, survival_times)
                    print("This is the calculated loss", loss)

                    running_losses += loss.item() * inputs.size(0) # in case the last batch has fewer samples

                # Report and print epoch losses
                avg_loss = running_losses / len(valid_dataloader.dataset)
                print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
                epoch_validated.append(epoch+1)
                epoch_valid_losses.append(avg_loss)

                print("epoch_validated", epoch_validated)
                print("epoch_valid_losses", epoch_valid_losses)
                plot_valid_loss_curves(loss_plot_out_dir, epoch_validated, epoch_valid_losses) # plot loss curves after each epoch

            print(f"--Validation finished for epoch {epoch+1}--")
        
        # Clear GPU cache
        torch.cuda.empty_cache()

    print("Training Finished!")
