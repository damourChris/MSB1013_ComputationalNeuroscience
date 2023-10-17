import numpy as np
import pylab as plt

import torch


def param_histogram(samples, labels=None, figsize=(10, 10)):
    """
    Create a histogram summary of parameters from samples.
    """
    num_samples, num_dims = samples.shape

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(num_dims, 1, figsize=figsize)
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    for i in range(num_dims):    
        ax[i, 1].hist(samples[:, i], bins=50, density=True, histtype="step", color="black")
        ax[i, 1].set_xlabel(labels[i])
        ax[i, 1].set_ylabel(labels[i])
        ax[i, 1].set_yticks([])
    return fig, ax



def pairplot(samples, labels=None, figsize=(10, 10)):
    """
    Create a pairplot from samples.
    """
    num_samples, num_dims = samples.shape

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(num_dims, num_dims, figsize=figsize)
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