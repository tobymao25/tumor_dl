import ray
import tune
from torch.utils.data import DataLoader
from image_branch_utils import GBMdataset
from image_branch_train import run_hyperparameter_search

if __name__ =='__main__':

    # Define the directory with images and segmentation
    image_dir = "./TrainingDataset/images"

    # Define the path to the CSV file containing the patient survival data
    csv_path = "./TrainingDataset/survival_data.csv"

    # Create the dataset
    dataset = GBMdataset(image_dir=image_dir, csv_path=csv_path)

    #Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("data loader created")
    1/0

    # input_shape = (1, 64, 64, 64)  # Example shape
    # network_depth = 4
    # no_convolutions = 2
    # conv_filter_no_init = 16
    # conv_kernel_size = 3
    # latent_representation_dim = 128
    # dropout_value = 0.3
    # use_batch_normalization = True
    # activation = 'relu'
    # l1 = 0.0
    # l2 = 0.0
    # lr = 1e-3

    search_space = {
    "input_shape": (5, 128, 128, 128),  # Fixed for all searches
    "network_depth": tune.choice([2, 3, 4, 5]),
    "no_convolutions": tune.choice([1, 2, 3]),
    "conv_filter_no_init": tune.choice([8, 16, 32, 64]),
    "conv_kernel_size": tune.choice([3, 5]),
    "latent_representation_dim": tune.choice([64, 128, 256]),
    "dropout_value": tune.uniform(0.2, 0.4),
    "use_batch_normalization": tune.choice([True, False]),
    "activation": tune.choice(["relu", "leakyrelu"]),
    "lr": tune.loguniform(1e-4, 1e-3),
    "epochs": tune.choice([10, 20, 30])
    }

    ray.init()
    run_hyperparameter_search()
