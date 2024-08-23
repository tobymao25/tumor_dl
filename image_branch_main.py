import ray
import ray.tune as tune
from image_branch_train import train_model, run_hyperparameter_search

if __name__ =='__main__':

    # config = {
    # 'input_shape': (5, 128, 128, 128), 
    # 'network_depth': 4, 
    # 'no_convolutions': 2, 
    # 'conv_filter_no_init': 8, #16, 
    # 'conv_kernel_size': 3, 
    # 'latent_representation_dim': 64, #128, 
    # 'dropout_value': 0.2, 
    # 'use_batch_normalization': True, 
    # 'activation': "relu", 
    # "lr": 1e-4, 
    # "epochs": 10
    # }

    # search_space = { #l1, l2???
    # "input_shape": (5, 128, 128, 128),  # Fixed for all searches
    # "network_depth": tune.choice([4]), 
    # "no_convolutions": tune.choice([2]), 
    # "conv_filter_no_init": tune.choice([8]), 
    # "conv_kernel_size": tune.choice([3]),
    # "latent_representation_dim": tune.choice([64]), 
    # "dropout_value": tune.choice([0.2]),
    # "use_batch_normalization": tune.choice([True]), 
    # "activation": tune.choice(["relu"]),
    # "lr": 1e-4,
    # "epochs": tune.choice([10])
    # }

    search_space = { #l1, l2???
    "input_shape": (5, 128, 128, 128),  # Fixed
    "network_depth": tune.choice([2, 4]), # Only supports 2 or 4
    "no_convolutions": tune.choice([1, 2, 3]),
    "conv_filter_no_init": tune.choice([8]),
    "conv_kernel_size": tune.choice([3, 5]),
    "latent_representation_dim": tune.choice([64, 128, 256]), 
    "dropout_value": tune.uniform(0.2, 0.4),
    "use_batch_normalization": tune.choice([True, False]),
    "activation": tune.choice(["relu", "leakyrelu"]),
    "lr": tune.loguniform(1e-4, 1e-3),
    "epochs": tune.choice([10, 20, 30])
    }

    ray.init()
    run_hyperparameter_search(search_space=search_space)
