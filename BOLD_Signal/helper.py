import numpy as np
from scipy import signal


def combine_inh_exc_abs_sum(Y, layers=4):
    """Inhibitory and excitatory input are combined per layer as sum of the absolute values.
    This function is able to deal with either a 3-dimensional array consisting of multiple simulations or a
    2-dimensional array of one simulation

    This sum is returned as 4-layer neural signal
    """
    def combine(curr_y):
        combined = np.zeros((curr_y.shape[0], layers))
        for layer in range(layers):
            combined[:, layer] = np.abs(curr_y[:, 2*layer]) + np.abs(curr_y[:,2*layer+1])
        return combined

    if len(Y.shape) == 3:  # multiple simulations
        combined_Ys = np.zeros((Y.shape[0], Y.shape[1], layers))
        for i, curr_y in enumerate(Y):
            combined_y = combine(curr_y)
            combined_Ys[i] = combined_y
        return combined_Ys
    else:  # single simulation
        return combine(Y)


def combine_inh_exc_only_exc(Y, layers=4):
    """This method just returns the excitatory input from the neural activity

    The returned neural activity is a 4-layer neural signal
    """
    def combine(curr_y):
        combined = np.zeros((curr_y.shape[0], layers))
        for layer, exc_index in zip(range(0, 4), range(0, 2*layers, 2)):
            combined[:, layer] = curr_y[:, exc_index]
        return combined

    if len(Y.shape) == 3:  # multiple simulations
        combined_Ys = np.zeros((Y.shape[0], Y.shape[1], layers))
        for i, curr_y in enumerate(Y):
            combined_y = combine(curr_y)
            combined_Ys[i] = combined_y
        return combined_Ys
    else:  # single simulation
        return combine(Y)


def downsample_neural_activity(neural_signal, original_sample_rate=1e-4, target_sample_rate=0.001):
    """The neural simulation given has a sampling rate of 1e-4 by default. For creating the bold signal, we need to
    down-sample accordingly to a realistic fMRI sample rate

    :param neural_signal: the simulated neural signal
    :param original_sample_rate: The sampling rate of the simulation
    :param target_sample_rate: The target sampling rate that should be used for bold

    :returns: down-sampled neural signal
    """
    # TODO: think about downsampling method, but in the example we just take the time points less frequently without
    #  doing mathematical downsampling
    resampling_factor = int(target_sample_rate/original_sample_rate)  # naive assumptions that this works
    resampled_signal = neural_signal[::resampling_factor]
    return resampled_signal
