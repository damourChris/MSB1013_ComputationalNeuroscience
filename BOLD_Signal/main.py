import os
from pathlib import Path

import numpy as np

from BOLD_Signal.helper import get_betas_from_neural_activity, plot_neural_activity_and_betas, \
    plot_neural_activity_and_bold


def get_simulations(file_name="simulations.npy"):
    """We assume we saved the simulations in a subfolder data of the bold folder

    We read this file and return it as a numpy array
    """
    data_path = os.path.join(Path(__file__).parent, "data", file_name)

    simulations_without_baseline = np.load(data_path)

    simulation_sampling_rate = 1e-4

    baseline_duration_added = 10
    baseline_duration_steps = int(baseline_duration_added / simulation_sampling_rate)
    baseline = np.load(os.path.join(Path(__file__).parent.parent, "Neurons_Simulations", "baseline.npy"))
    baseline = baseline[:baseline_duration_steps, :]

    batch_size = simulations_without_baseline.shape[0]

    simulations_baseline = np.tile(baseline, (batch_size, 1, 1))

    simulations_with_baseline = np.concatenate([simulations_without_baseline, simulations_baseline], axis=1)
    return simulations_with_baseline


def write_betas_for_batch(file_name):
    print(f"Create betas for batch with file name {file_name}")
    simulations = get_simulations(file_name=file_name)
    print(f"Running beta retrieval for simulation with shape {simulations.shape}")
    betas = np.zeros(shape=(simulations.shape[0], 4))
    for i in range(simulations.shape[0]):
        print(f"Current iteration: {i}")
        neural_activity = simulations[i, :, :]
        B, X, bold_responses = get_betas_from_neural_activity(
            neural_activity,
            neural_activity_sampling_rate=1e-4,
            bold_sample_rate=0.001,
            stim_start=0.5,
            stim_end=2.5
        )
        betas[i, :] = B
        if i == 0:
            plot_neural_activity_and_betas(neural_activity, B, X)
            plot_neural_activity_and_bold(neural_activity, bold_responses)
    betas_filename = file_name.replace("Y_", "Betas_")
    betas_path = os.path.join(Path(__file__).parent, "data", betas_filename)
    np.save(betas_path, betas)


if __name__ == '__main__':
    file_name = "Y_01.npy"
    write_betas_for_batch(file_name)
