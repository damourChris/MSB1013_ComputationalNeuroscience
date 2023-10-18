import numpy as np
import pylab as plt

import math
import torch


def percentile_distribution(samples, labels=None, figsize=(10, 10), percentiles=[5,95], ylims=[-1000,1000]):
    """
    Create a histogram summary of parameters from samples.
    """
    num_samples, num_dims = samples.shape
    plot_dims = math.ceil(math.sqrt(num_dims))

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]
    
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots(plot_dims, plot_dims, figsize=figsize)
    
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    
    p = np.linspace(percentiles[0], percentiles[1], 1000)
    
    for i in range(plot_dims):    
        for j in range(plot_dims): 
            indx = i + j*plot_dims 
            if(indx < num_dims):
                mat = np.matrix(samples[:, indx])
                percentiles = np.percentile(samples[:, indx], p)
                ax[j, i].plot(p, percentiles)
                ax[j, i].plot(p, np.repeat(0, np.shape(p)))
                ax[j, i].set_xlabel(labels[indx])
                ax[j, i].set_ylabel(labels[indx])
                
                
                ax[j, i].set_ylim(ylims)
                # ax[j, i].set_yticks()
            else:
                ax[j, i].axis("off")
    return fig, ax



def pairplot(samples, labels=None, figsize=(10, 10)):
    """
    Create a pairplot from samples.
    """
    num_samples, num_dims = samples.shape

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(num_dims, num_dims, figsize=figsize, layout='constrained')
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    for i in range(num_dims):
        for j in range(num_dims):
            if (i == j):
                ax[i, j].hist(samples[:, i], bins=50, density=True, histtype="step", color="black")
                ax[i, j].set_xlabel(labels[j])
                ax[i, j].set_ylabel(labels[i])
                ax[i, j].set_yticks([])
            if (i < j):
                ax[i, j].hist2d(samples[:, j], samples[:, i], bins=50, cmap="Reds")
                ax[i, j].set_xlabel(labels[j])
                ax[i, j].set_ylabel(labels[i])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if (i > j):
                ax[i, j].axis("off")

    return fig, ax

def marginal_correlation(samples, labels=None, figsize=(10, 10)):
    """
    Create a marginal correlation matrix.
    """

    num_samples, num_dims = samples.shape

    corr_matrix_marginal = np.corrcoef(samples.T)

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    ax.set_xticks(np.arange(num_dims))
    ax.set_yticks(np.arange(num_dims))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    _ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax