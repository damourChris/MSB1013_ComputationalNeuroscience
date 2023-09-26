import numpy as np
import pylab as plt
import os

import torch
from sbi.inference import SNPE, SNLE, SNRE

def train(num_simulations,
          x, 
          theta,
          num_threads = 1,
          method = "SNPE",
          device = "cpu",
          density_estimator = "maf"
         ):
    
    torch.set_num_threads(num_threads)

    if (len(x.shape) == 1):
        x = x[:, None]
    if (len(theta.shape) == 1):
        theta = theta[:, None]
    
    # Function to handle unknown methods
    def unknown_method():
        # Raise a ValueError if a method is not known
        raise ValueError("Unknown inference method")

    # Create a dictionary that maps method names to their respective classes
    methods = {
        "SNPE": SNPE,
        "SNLE": SNLE,
        "SNRE": SNRE
    }

    try:
        # Get the class for the given method from the dictionary
        # If the method is not found in the dictionary, call the unknown_method function
        InferenceMethod = methods.get(method, unknown_method)

        # Instantiate the class with the given parameters
        inference = InferenceMethod(density_estimator=density_estimator, device=device)
    
    except ValueError as e:
        # Print the error message if a ValueError is raised
        print(e)

    
    # Once the inference method has been initialize we can append simulation data to it
    inference = inference.append_simulations(theta, x)
    
    # Then we can train the neural network 
    _density_estimator = inference.train()
    
    # Once training is done, build the posterior and return it 
    posterior = inference.build_posterior(_density_estimator)
    
    return posterior

def infer(obs_stats,
          num_samples,
          posterior):
    return posterior.sample((num_samples,), x=obs_stats)


if __name__=="__main__":

    import argparse 

    parser = argparse.ArgumentParser(
        description = "Train a density estimator on the Lorenz system.")
    
    parser.add_argument(
        "--data",
        type = str,
        default = "data/X.npy",
        help = "Path to the data file."
    )
    
    parser.add_argument(
        "--method",
        type = str,
        default = "SNPE",
        help = "Inference method."
    )
    
    parser.add_argument(
        "--density_estimator",
        type = str,
        default = "maf",
        help = "Density estimator."
    )
    
    parser.add_argument(
        "--num_threads",
        type = int,
        default = 1,
        help = "Number of threads."
    )
    
    parser.add_argument(
        "--device",
        type = str,
        default = "cpu",
        help = "Device."
    )
    
    parser.add_argument(
        "--model_dir",
        type = str,
        default = "models",
        help = "Define the directory of where to save the model."
    )
    
    parser.add_argument(
        "--posterior_filename",
        type = str,
        default = "posterior.pt",
        help = "Define the filename of the posterior model."
    )
    

    # Once the parser has all the arguments we can parse into the args object
    args = parser.parse_args()

    # Load the data with the --data argument into a matrix 
    X = np.load(args.data, allow_pickle=True)

    # Seperate simulation parameters and summary statistics
    # params = X[:, -3:] # | NEED TO CHANGE TO WORK WITH OUR DATA
    # stats  = X[:, :-3] # |

    # Establish how many simulation are in the data 
    num_simulations = X.shape[0]

    # When working with Torch, the matrix has to be parsed to a Torch object 
    theta = torch.from_numpy(params).float()
    x = torch.from_numpy(stats).float()

    print("Training posterior model...")
    
    # Train the posterior with all the arguments needed 
    posterior = train(num_simulations,
                        x,
                        theta,
                        num_threads         = args.num_threads,
                        method              = args.method,
                        device              = args.device,
                        density_estimator   = args.density_estimator
                        )
    
    print("Training finished. Saving model...")
    
    # Check if model directory exist, if not create it.
    if not os.path.exists(args.models_dir)
        try:
            os.makedirs(path)
            print("Model Directory: ", args.models_dir, " Created")
            
        except OSError as e:
            
            print("Input Model Directory could not be created.")
            
    
    model_path = os.path.join(args.models_dir, args.posterior_filename)
    torch.save(posterior, model_path)
    print("Posterior model: ", model_path, " saved")
    
    