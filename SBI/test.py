import numpy as np

from tqdm import tqdm

import sys

def test_posterior(
        posterior, 
        test_matrix,
        num_layers = 8,
        num_betas = 4,
        num_samples = 10000,
        threshold = 0.2, 
        tolerance = 0.5,
        activation_threshold = 5.0,
        percentile = 50
    ):
    """
        A method to generate statistics for a model. It takes a trained model and a matrix of testing data. 
    """
    
    num_tests = test_matrix.shape[0]
    
    test_results = {}
    test_results['value_in_range'] = np.zeros((num_tests, num_layers))
    test_results['percentile'] = np.zeros((num_tests, num_layers))
    test_results['activated_layers'] = np.zeros((num_tests, num_layers))
    test_results['true_layers'] = np.zeros((num_tests, num_layers))
    test_results['predictions_binary'] = np.zeros((num_tests, num_layers))
    test_results['prediction_sucess_binary'] =  np.zeros((num_tests))
    test_results['std_per_layer'] =  np.zeros((num_tests, num_layers))
    test_results['mean_per_layer'] =  np.zeros((num_tests, num_layers))
    
    # Initialize array for calculating percentile values
    p = np.linspace(0,100, 1000)
    
    # loop over all the testing data 
    for cur_test_indx in tqdm(range(0, num_tests), unit=" tests", colour="green", file=sys.stdout,position=0):
        
        # Get default stats
        obs_x = test_matrix[cur_test_indx, range(num_layers,num_layers+num_betas) ] 
        
        # Get default params
        obs_theta = test_matrix[cur_test_indx, range(num_layers)] 
        
        posterior.set_default_x(obs_x)
        posterior_samples = posterior.sample((num_samples,), show_progress_bars=False)
        
        # Initialize temporary arrays 
        in_bounds = np.zeros((num_layers))
        percentile_value = np.zeros((num_layers))
        means = np.zeros((num_layers))
        stds = np.zeros((num_layers))
        
        # Get stats for each layer
        for cur_layer in range(num_layers):    
            
            true_value = obs_theta[cur_layer]
            layer_samples = posterior_samples[:, cur_layer]
            
            means[cur_layer] = np.mean(np.array(layer_samples))
            stds[cur_layer] = np.std(np.array(layer_samples))
            
            # Get percentile value and values for each percentile
            # p is array of 0 -> 100 so that percentiles has the all the values for the percentiles
            percentile_value[cur_layer] = np.percentile(layer_samples, percentile)
            percentiles = np.percentile(layer_samples, p)
            
            # Establish range for prediction compared to true value
            lower_bound = true_value*(1 - threshold) - tolerance
            upper_bound = true_value*(1 + threshold) + tolerance
            
            # Calculate how much of the samples lies in the established range
            percentiles_in_range = [p[i] for i in range(p.shape[0]) if lower_bound <= percentiles[i] < upper_bound]
            
            in_bounds[cur_layer] = max(percentiles_in_range)-min(percentiles_in_range) if len(percentiles_in_range) > 0 else 0
    
        # Get binary vector indicating wich layers were activated (obs_theta)

        true_activated_layers = np.array([1 if obs_theta[i] > 0.1 else 0 for i in range(num_layers)])
   
        # Assign 1 if the number of samples with a specific layer counter as activated is bigger than activation_thresold
        predicted_actived_layers = np.array([1 if np.percentile(posterior_samples[:,i], percentile) > activation_threshold else 0 for i in range(num_layers)])
        
        # Store testing results 
        test_results['std_per_layer'][cur_test_indx]  =  means
        test_results['mean_per_layer'][cur_test_indx]  =  stds
        test_results['true_layers'][cur_test_indx]      = true_activated_layers
        test_results['activated_layers'][cur_test_indx] = predicted_actived_layers
        test_results['predictions_binary'][cur_test_indx]= (true_activated_layers == predicted_actived_layers).astype(int)
        test_results['percentile'][cur_test_indx]       = percentile_value
        test_results['value_in_range'][cur_test_indx]   = in_bounds
    
    return test_results

def return_single_results(test_results, index):
    res = {}
    for key in  test_results.keys():
        res[key] = test_results[key][index]
    return res

def print_stats(test_result):
    print("Prediction of active layers:  ", list(np.where(np.array(list(map(lambda x: round(x,2), map(float, test_result['activated_layers'])))) > 0 ))[0])
    print("True actives layers:          ", list(np.where(test_result['true_layers'] > 0))[0])
    print("Layer correctly predicted:    ", list(np.where(test_result['predictions_binary'] > 0))[0])
    print("Number of predicted layers:   ", round(sum(test_result['predictions_binary'])))
    print("Predicted value (50th %tile): ", list(map(lambda x: round(x,2), map(float,test_result['percentile']))))
    print("Percentage of sample in range:", list(map(lambda x: round(x,2), map(float,test_result['value_in_range']))))