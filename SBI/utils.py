import numpy as np
import pylab as plt

from itertools import combinations

from test import balanced_accuracy_single, balanced_accuracy_double

import math
import torch


def check_array_length(arr1, arr2, custom_msg=None):
    if len(arr1) != len(arr2):
        if custom_msg is None:
            raise ValueError("Arrays are not of the same length")
        else:
            raise ValueError(custom_msg)

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


def accuracy_per_layer(binary_layer_test_results,
                       activated_layers,
        num_layers = 8,
       figsize=(10,5)
        ):
    
    num_tests = len(binary_layer_test_results)
    combination_success = np.zeros((num_layers+1,num_layers+1))
    single_layer_success = np.zeros((num_layers))
    activated_layers_ratio =  np.zeros((num_layers))

    all_1_combis_keys = list(combinations(range(num_layers), 1)) 
    all_2_combis_keys = list(combinations(range(num_layers), 2))

    layers_1_combinations_num = dict.fromkeys(all_1_combis_keys, [0])
    layers_2_combinations_num = dict.fromkeys(all_2_combis_keys, [0])

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    correctly_predicted_layers = list(map(lambda x: np.where(x > 0), binary_layer_test_results))
    
    for val in all_1_combis_keys:
        activated_layers_ratio[val[0]] = sum([1 if activated_layers[i][val] == 0 else 0 for i in range(num_tests)])/num_tests
        single_layer_success[val[0]] = sum([1 if sum(np.isin(val,correctly_predicted_layers[i][0])) == 1 else 0 for i in range(num_tests)])/num_tests

    ax[0].bar(list(range(1,num_layers+1)), single_layer_success)
    ax[0].bar(list(range(1,num_layers+1)), activated_layers_ratio, alpha=0.8)
    ax[0].set_xticks(list(range(1,num_layers+1)))
    ax[0].set_ylabel("Ratio of accurate predictions")
    ax[0].set_xlabel("Layer")
    ax[0].set_ylim([0,1])
    # plt.title("Prediction Success per layer")
    ax[0].set_title("Correct prediction per single layer")


    for val in all_2_combis_keys:
        combination_success[val[0]+1,val[1]+1] = sum([1 if sum(np.isin(val,correctly_predicted_layers[i][0])) == 2 else 0 for i in range(num_tests)])/num_tests
    for (j,i), value in np.ndenumerate(combination_success):
        if(value > 0):
            plt.text(i, j, round(value,2), ha='center', va='center')
    ax[1].set_title("Correct prediction per double layer")
    ax[1].imshow(combination_success, cmap="PiYG")
    ax[1].set_xticks(list(range(1,num_layers+1)))
    ax[1].set_yticks(list(range(1,num_layers+1)))
    ax[1].set_xlim([0.5,num_layers+0.5])
    ax[1].set_ylim([0.5,num_layers+0.5])
    
    return fig, ax


def plot_confusion_matrices_and_balanced_accuracies(
        test_result,
        population_names = ["$1_e$","$1_i$","$2_e$","$2_i$","$3_e$","$3_i$","$4_e$","$4_i$"]
    ):
    # Read results 
    pred_layers = test_result['predicted_layers']
    true_layers = test_result['true_layers']
    
    num_layers = len(pred_layers[0]) 
    
    # Calculate how many rows/cols for heatmap
    num_dims = int(np.ceil(np.sqrt(num_layers)))

    # Setup main plots
    fig = plt.figure(layout='constrained', figsize = (18, 9))
    subfigs = fig.subfigures(1, 2, wspace = 0.07)

    # Setup suplots
    axLeft  = subfigs[0].subplots(num_dims, num_dims)
    axRight = subfigs[1].subplots(       1,        1)

    # confusion matrix labels
    confusion_matrix_labels = [['True Negative',' False Negative'],['False Positive',' True Positive']]

    # Calculate balanced accuracy 
    balanced_accuracy, single_layer_success = balanced_accuracy_single(pred_layers, true_layers)
    
    images = []

    for i in range(num_dims):
        for j in range(num_dims):

            # calculate current 1-D index
            indx = i + j*num_dims

            if(indx < num_layers):

                # get data for current confusion matrix
                heatmap_data = single_layer_success[indx]

                # Plot heatmap of True/False positive/negative
                im = axLeft[j,i].imshow(heatmap_data, cmap = "coolwarm")

                # keep track of images (for colorbar)
                images.append(im)

                # Turn y labels vertically 
                axLeft[j,i].tick_params(axis = "y", labelrotation = 90)

                # Y axis
                axLeft[j,i].set_yticks(list(range(2)))
                axLeft[j,i].set_yticklabels(['Negative', 'Positive'])
                axLeft[j,i].set_ylabel("Truth")

                # X axis
                axLeft[j,i].set_xticks(list(range(2)))
                axLeft[j,i].set_xticklabels(['Negative', 'Positive'])
                axLeft[j,i].set_xlabel("Prediction")

                # Title 
                axLeft[j,i].set_title("Population:" + population_names[indx] , fontsize=10)

                # Plot values directly on heatmap for easier comprehension
                for (k,l), value in np.ndenumerate(heatmap_data):
                    axLeft[j,i].text(l, k     , round(value,3)                , ha = 'center', va = 'center', fontsize = 20)
                    axLeft[j,i].text(l, k+0.25, confusion_matrix_labels[k][l] , ha = 'center', va = 'center', fontsize = 8)
            else: 
                axLeft[j,i].axis('off')

    subfigs[0].colorbar(images[0], ax=axLeft, orientation='horizontal')
    
    
    # Plot all the balanced accuracy measure for each population             
    axRight.bar(list(range(len(pred_layers[0]) )), list(balanced_accuracy))

    # Y axis
    axRight.set_ylim([0,1])
    axRight.set_yticks(np.arange(0,1.05,0.05))
    axRight.set_ylabel("Balanced Accuracy")

    # X axis
    axRight.set_xlabel("Population")
    axRight.set_xticklabels([''] + population_names)

    

    fig.suptitle("Confusion Matrices and Balanced Accuracy")

    return fig

def plot_confusion_matrices_and_balanced_accuracies_combinations(
    test_result,
    population_names = ["$1_e$","$1_i$","$2_e$","$2_i$","$3_e$","$3_i$","$4_e$","$4_i$"]
    ):

    # Read results 
    pred_layers = test_result['predicted_layers']
    true_layers = test_result['true_layers']

    # Get parameters
    num_tests = len(pred_layers)
    num_layers = len(pred_layers[0]) 

    # Setup Figures 
    fig = plt.figure(layout='constrained', figsize=(20, 10))
    subfigs = fig.subfigures(1,2, wspace=0.02)

    # Initialize sub figures
    axLeft  = subfigs[0].subplots(num_layers, num_layers)
    axRight = subfigs[1].subplots(1,1) 

    

    # Generate all 2-combinations of populations
    combination_labels      = list(combinations(population_names, 2))

    # confusion matrix labels
    confusion_matrix_labels = [['True Negative',' False Negative'],['False Positive',' True Positive']]

    combination_success, balanced_accuracy = balanced_accuracy_double(pred_layers, true_layers)

    # Plot results from calcuations

    # setup indx to keep track of confusion matrix label 
    indx = 0
    images = []

    for i in range(num_layers):
        for j in range(num_layers):

            # Check if the cur plot(combination) has data associated if not turn off axis
            #   note that this is not the most efficient since all the matrices are square to make life a bit easier so 
            #   matrix ends up being lower-triangular -> this check is easy way of removing the extra unnescary plots
            if(sum(sum(combination_success[(i,j)])) > 0):

                # get data for current confusion matrix
                heatmap_data = combination_success[(i,j)]

                # Plot heatmap of True/False positive/negative
                im = axLeft[j,i].imshow(heatmap_data, cmap="coolwarm")

                # keep track of images (for colorbar)
                images.append(im)


                # Turn y labels vertically 
                axLeft[j,i].tick_params(axis="y", labelrotation=90)

                # Y axis
                axLeft[j,i].set_yticks(list(range(2)))
                axLeft[j,i].set_yticklabels(['Negative', 'Positive'], fontsize = 4)
                axLeft[j,i].set_ylabel("Truth", fontsize = 6)

                # X axis
                axLeft[j,i].set_xticks(list(range(2)))
                axLeft[j,i].set_xticklabels(['Negative', 'Positive'], fontsize = 4)
                axLeft[j,i].set_xlabel("Prediction", fontsize = 6)

                # Title 
                axLeft[j,i].set_title(combination_labels[indx][0] + " | " + combination_labels[indx][1] , fontsize=10)
                indx += 1

                # Plot values directly on heatmap for easier comprehension
                for (k,l), value in np.ndenumerate(heatmap_data):
                    axLeft[j,i].text(l, k, round(value,2), ha='center', va='center', fontsize=8)
                    axLeft[j,i].text(l, k+0.25, confusion_matrix_labels[k][l], ha='center', va='center', fontsize=4)

            else: 
                axLeft[j,i].axis('off')

    subfigs[0].colorbar(images[0], ax=axLeft, orientation='vertical')
    
    # Plot balanced accuracy for each combinaiton of population
    axRight.imshow(balanced_accuracy, cmap="coolwarm")

    # Put text values
    for (k,l), value in np.ndenumerate(balanced_accuracy):

        # same check as earlier, since matrix is lower-triangular this check get makes sure we dont plot the extra zeros
        if value > 0: 
            axRight.text(l, k, round(value,3), ha='center', va='center', fontsize=17)                

    # weird matplotlib magic where the labels are one behind, 
    # so just add buffer label to allign the labels to the heat map
    # ???

    # Y axis
    axRight.set_yticklabels([''] + population_names, fontsize = 18)
    axRight.set_ylim([-.5,7.5])

    # X axis
    axRight.set_xticklabels([''] + population_names, fontsize = 18)

    axRight.set_xlim([-.5,7.5])
    axRight.invert_xaxis()


    # Titles
    subfigs[0].suptitle("Confusion Matrices", fontsize = 22)
    subfigs[1].suptitle("Balanced Accuracy per combination", fontsize = 22)
    fig.suptitle("Confusion Matrices and Balanced Accuracy double population detection", fontsize = 30)

    return fig

def compute_average_accuracy(test_results):
    
    overall_accuracy = np.zeros((len(test_results) ))
    
    for index, results in enumerate(test_results):
        
        balanced_accuracy , _ = balanced_accuracy_single(results['predicted_layers'], results['true_layers'])
        overall_accuracy[index] = sum(balanced_accuracy)/len(balanced_accuracy)
    
    return overall_accuracy

def plot_average_accuracy(test_results, figsize = (10,10) ):
    fig, ax = plt.subplots(1 , 1, figsize = figsize)
    general_accuracy = compute_average_accuracy(test_results_arr)
    
    ax.set_ylim([0,1])
    ax.set_ylabel("Balanced Accuracy")
    
    ax.set_xticks(range(1,len(test_results)+1))
    ax.set_xlabel("Model")
    
    ax.set_title("Accuracy per Model")
        
    ax.bar(range(1,len(test_results)+1), general_accuracy)
    
    return fig, ax

def plot_stats(test_result, figsize=(20,20)):
    fig = plt.figure(layout='constrained', figsize=(20, 10))
    subfigs = fig.subfigures(1,2, wspace=0.02)           
    
    ax11 = subfigs[0].subplots(2,2)
               
    ax11[0,0].bar(range(8), test_result['predicted_layers'])
    ax11[0,0].set_xlabel("Population")
    ax11[0,0].set_title("Binary Prediction")
    
    ax11[0,1].bar(range(8), test_result['true_layers'])
    ax11[0,1].set_xlabel("Population")
    ax11[0,1].set_title("Truth")
    subfigs[0].suptitle("Population Activation Prediction")
    
    
    # ax12 = subfigs[0,1].subplots(2,1)
               
    ax11[1,0].bar(range(8), test_result['predicted_values'])
    ax11[1,0].set_ylim([0,500])
    ax11[1,0].set_title("Value Prediction")
    ax11[1,0].set_xlabel("Population")
    
    ax11[1,1].bar(range(8), test_result['true_values'])
    ax11[1,1].set_ylim([0,500])
    ax11[1,1].set_title("Truth")
    ax11[1,1].set_xlabel("Population")
    
    # subfigs[0,1].suptitle("Population Activation Level Prediction")
    
    
    ax21 = subfigs[1].subplots(2,1)
               
    ax21[0].bar(range(8), test_result['values_in_range'])
    ax21[0].set_title("Percentage of samples in range of true value")
    
    ax21[1].bar(range(8), test_result['mean_per_layer'], yerr=test_result['std_per_layer'])
    ax21[1].set_title("Average of sample value")
    
    return fig

def get_layers_with_pop_active(test_result, pop):
    return np.where(np.array([1 if test_result['true_layers'][i][pop] > 0 else 0 for i in range(len(test_result['true_layers']))]) > 0)[0]