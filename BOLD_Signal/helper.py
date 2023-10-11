import numpy as np
from matplotlib import pyplot as plt

from BOLD_Signal.BWM import balloon_windkessel
from BOLD_Signal.DMF_model import HRF


def combine_inh_exc_abs_sum(Y, layers=4):
    """Inhibitory and excitatory input are combined per layer as sum of the absolute values.
    This function is able to deal with either a 3-dimensional array consisting of multiple simulations or a
    2-dimensional array of one simulation

    This sum is returned as 4-layer neural signal
    """

    def combine(curr_y):
        combined = np.zeros((curr_y.shape[0], layers))
        for layer in range(layers):
            combined[:, layer] = np.abs(curr_y[:, 2 * layer]) + np.abs(curr_y[:, 2 * layer + 1])
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
        for layer, exc_index in zip(range(0, 4), range(0, 2 * layers, 2)):
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
    resampling_factor = int(target_sample_rate / original_sample_rate)  # naive assumptions that this works
    resampled_signal = neural_signal[::resampling_factor]
    return resampled_signal


def remove_baseline_neural_activity(Y, stim_start=10, stim_end=20, sampling_rate=1e-4):
    """Remove the baseline neural activity by calculating the mean of the neural signal outside the stimulation interval
    and subtracting it from the whole signal
    """
    start_index = int(stim_start / sampling_rate)
    end_index = int(stim_end / sampling_rate)
    non_stimulation_indices = [i for i in range(Y.shape[0]) if i < start_index or i >= end_index]
    for layer in range(Y.shape[1]):
        mean_value = np.mean(Y[non_stimulation_indices, layer])
        Y[:, layer] = Y[:, layer] - mean_value
    return Y


def get_betas_from_neural_activity(Y, neural_activity_sampling_rate=1e-4, bold_sample_rate=0.001,
                                   stim_start=10, stim_end=20):
    """For a single simulation
    TODO: extend to make it work on whole array
    """
    TR = 2  # interval between MRI scan acquisitions in seconds
    simulation_time = Y.shape[0] * neural_activity_sampling_rate

    # at what indices we need to sample the downsampled neural activity and the bold response
    sampling_indices = np.arange(0, int(simulation_time / bold_sample_rate), int(TR / bold_sample_rate))

    # reduce to neural activity per layer
    neural_activity = combine_inh_exc_only_exc(Y)

    # remove baseline neural activity
    neural_activity = remove_baseline_neural_activity(neural_activity, stim_start, stim_end,
                                                      neural_activity_sampling_rate)

    # down-sample to match dt of bold
    neural_activity = downsample_neural_activity(neural_activity)

    # bold responses for the layers
    bold_downsampled = np.zeros(shape=(sampling_indices.shape[0], neural_activity.shape[1]))
    bold_responses = np.zeros(shape=neural_activity.shape)

    X = np.zeros((Y.shape[0], 4))

    for layer in range(neural_activity.shape[1]):
        bold, f, v, q = balloon_windkessel(neural_activity[:, layer])
        bold_responses[:, layer] = bold

        # sample bold with TR
        bold_response = bold[sampling_indices]
        bold_downsampled[:, layer] = bold_response

        hrf = HRF(np.arange(0, 40, 1e-3))

        # create condition vector for GLM analysis
        condition = np.zeros((Y.shape[0]))
        condition[int(stim_start/neural_activity_sampling_rate): int(stim_end/neural_activity_sampling_rate)] = 1

        # predicted BOLD signal
        currX = np.convolve(condition, hrf)[:len(condition)]
        # scale between 0 and 1
        currX = (currX - np.min(currX)) / (np.max(currX) - np.min(currX))

        X[:, layer] = currX

    # down-sampled X at fMRI scan acquisition points
    X = X[::int(TR/neural_activity_sampling_rate), :]
    plt.plot(X[:, 0])

    # scale betas to obtain original signal (Y = X*B)
    B = (np.linalg.pinv(X @ X.T) @ X).T @ bold_downsampled
    B = B[0, :]
    return B, X, bold_downsampled
