import numpy as np
import pylab as plt

from itertools import combinations


import math
import torch


def percentile_distribution(samples, labels=None, figsize=(10, 10), percentiles=[5,95], ylims=[-1000,1000], return_percentile=False):
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
    
    if(return_percentile):
        percentile_values = np.empty(8)
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
                if(return_percentile):
                    percentile_values[indx] = np.percentile(samples[:, indx], 50)
                
                ax[j, i].set_ylim(ylims)
                # ax[j, i].set_yticks()
            else:
                ax[j, i].axis("off")
    if(return_percentile):
        return fig, ax, percentile_values
    else:
        return fig, ax

def plot_layer_combinations(test_matrix, figsize = (10,10), num_layers = 8):
    
    layers_combinations_indices = {} 
    for index, value in enumerate([[j for j in range(8) if test_matrix[i, j] > 0] for i in range(test_matrix.shape[0])]):
        
        combination = tuple(value)
        
        if combination in layers_combinations_indices:
            layers_combinations_indices[combination].append(index)
        else:
            layers_combinations_indices[combination] = [index]

    all_1_combis_keys = list(combinations(range(num_layers), 1)) 
    all_2_combis_keys = list(combinations(range(num_layers), 2))
    
    layers_1_combinations_num = dict.fromkeys(all_1_combis_keys, [0])
    layers_2_combinations_num = dict.fromkeys(all_2_combis_keys, [0])
    for key in layers_combinations_indices:
        value = layers_combinations_indices[key]
        if key in layers_1_combinations_num:
            layers_1_combinations_num[key] = [len(value)]
        if key in layers_2_combinations_num:
            layers_2_combinations_num[key] = [len(value)]
            
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    # print()
    ax[0].bar(list(range(1,num_layers+1)), [x[0] for x in  layers_1_combinations_num.values()])
    ax[0].set_xticks(list(range(1,num_layers+1)))
    ax[0].set_ylabel("Number of simulations")
    ax[0].set_xlabel("Layer")
    # plt.title("Prediction Success per layer")
    ax[0].set_title("Number of Simulations with only 1 layer activated")
    
    heatmap_data = np.zeros((num_layers+1, num_layers+1))

    for key, value in layers_2_combinations_num.items():
        heatmap_data[key[0]+1][key[1]+1] = value[0]
    ax[1].imshow(heatmap_data, cmap="PiYG")
    ax[1].set_xticks(list(range(1,num_layers+1)))
    ax[1].set_yticks(list(range(1,num_layers+1)))
    ax[1].set_xlim([0.5,num_layers+0.5])
    ax[1].set_ylim([0.5,num_layers+0.5])
    ax[1].set_xlabel("Layer")
    ax[1].set_ylabel("Layer")
    for (j,i), value in np.ndenumerate(heatmap_data):
        if(int(value) > 0):
            plt.text(i, j, int(value), ha='center', va='center')
            
    ax[1].set_title("Number of Simulations with only 2 layers activated - Combinations")
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

