import logging
import ray
import ray.tune as tune
from image_branch_train import train_model, run_hyperparameter_search

if __name__ =='__main__':
    
    config = {
    'input_shape': (5, 128, 128, 128), 
    'network_depth': 4, 
    'no_convolutions': 2, 
    'conv_filter_no_init': 64, 
    'conv_kernel_size': 3, 
    'latent_representation_dim': 256,
    'dropout_value': 0.1, 
    'use_batch_normalization': False, 
    'activation': "relu", 
    "lr": 1e-4, 
    "epochs": 50
    }

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

    # search_space = { #l1, l2???
    # "input_shape": (5, 128, 128, 128),  # Fixed
    # "network_depth": tune.choice([2, 4]), # Only supports 2 or 4
    # "no_convolutions": tune.choice([2, 3]), #, 3
    # "conv_filter_no_init": tune.choice([8]), # TODO - 
    # "conv_kernel_size": tune.choice([3, 5]), #, 5
    # "latent_representation_dim": tune.choice([32, 64]), 
    # "dropout_value": tune.uniform(0.1, 0.2), #0.4
    # "use_batch_normalization": tune.choice([False]), #True
    # "activation": tune.choice(["relu", "leakyrelu"]),
    # "lr": tune.uniform(1e-5, 1e-4), 
    # "epochs": tune.randint(30, 100),  # likely > 50, previous paper used 2000
    # }

    #ray.init()
    #logging.getLogger("ray").setLevel(logging.INFO)
    #run_hyperparameter_search(search_space=search_space, num_samples=1)

    train_model(config=config)