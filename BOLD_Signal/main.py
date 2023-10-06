import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from BOLD_Signal.BWM import balloon_windkessel
from BOLD_Signal.helper import downsample_neural_activity, combine_inh_exc_only_exc


def get_simulations(file_name="simulations.npy"):
    """We assume we saved the simulations in a subfolder data of the bold folder

    We read this file and return it as a numpy array
    """
    data_path = os.path.join(Path(__file__).parent, "data", file_name)
    simulations = np.load(data_path)
    return simulations


if __name__ == '__main__':
    simulations = get_simulations(file_name="y_results_subset.npy")
    neural_activity = combine_inh_exc_only_exc(simulations[0, :, :])
    current_y = neural_activity[:, 1]
    down_sampled = downsample_neural_activity(neural_activity, original_sample_rate=1e-4, target_sample_rate=0.001)
    bold_signal, _, _, _ = balloon_windkessel(current_y)
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(current_y)
    plt.title('Neural Activity')
    plt.subplot(2, 1, 2)
    plt.title('BOLD response')
    plt.plot(bold_signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
