import numpy as np
import pylab as plt

import torch

# from sbi.analysis import conditional_pairplot
from utils import pairplot, marginal_correlation

def sample_posterior(
        obs_x,
        obs_theta,
        model = "models/posterior.pt",
        num_samples = 100000,
        save_plot= True,
        plot_dir = "plot_images"
    ):

    # mean, cov, cor, eigvals, eigvecs, lyap = statistics(x_t)
    
    # obs_x = np.concatenate([mean, cov.flatten(), cor.flatten()])
    # obs_theta = [sigma, beta, rho]

    # Load posterior. If model is a string, load with torch otherwise assume it is already a torch model 
    #   - ideally would want to check the type aswell and throw an error if the type doesnt match str or Torch model 
    if isinstance(model, str)
        posterior = torch.load(model)
    else 
        posterior = model
        
    # Set true value ? 
    posterior.set_default_x(obs_x)
    
    # Sample the posterior
    posterior_samples = posterior.sample((num_samples,))

    
    # Plot the results 
    fig, ax = pairplot(
        samples = posterior_samples,
        labels  = [r"$\sigma$", r"$\beta$", r"$\rho$"],
        figsize = (10, 10)
    )
    
    if save_plot:
        plt.savefig("png/pairplot.png")
    
    
    fig, ax = marginal_correlation(
        samples = posterior_samples,
        labels  = [r"$\sigma$", r"$\beta$", r"$\rho$"], 
        figsize = (10, 10)
    )

    if save_plot:
        plt.savefig("png/marginal_correlation.png")

    # import IPython; IPython.embed();