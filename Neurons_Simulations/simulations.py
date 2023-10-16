import os
import numpy as np
from DMF_model import DMF_sim, DMF_parameters


def get_result_folder():
    folder = "COMBIS"
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def get_combinations(create=False):
    # Define the possible values for each input intensity.
    valuesI = [50, 100, 150, 200, 250, 300]
    valuesLayer = range(8)
    folder_path = get_result_folder()
    if create:
        # Create an array of all possible combinations for each input
        input_combinations = np.array(np.meshgrid(valuesI, valuesI)).T.reshape(-1, 2)
        layer_combinations = np.array(np.meshgrid(valuesLayer, valuesLayer)).T.reshape(-1, 2)

        # repeat each input combination for each layer combination
        input_combinations = np.repeat(input_combinations, len(layer_combinations), axis=0)
        layer_combinations = np.tile(layer_combinations, (len(valuesI) ** 2, 1))

        # repeat each input combination and layer combination 5 times
        input_combinations = np.repeat(input_combinations, 5, axis=0)
        layer_combinations = np.repeat(layer_combinations, 5, axis=0)

        # Create random input combinations centered around the original input combinations
        input_combinations = input_combinations + np.random.uniform(-50, 49, input_combinations.shape)

        # save input and layer combinations
        np.save(os.path.join(folder_path, "input_combinations.npy"), input_combinations)
        np.save(os.path.join(folder_path, "layer_combinations.npy"), layer_combinations)
    else:
        input_combinations = np.load(os.path.join(folder_path, "input_combinations.npy"))
        layer_combinations = np.load(os.path.join(folder_path, "layer_combinations.npy"))
    return input_combinations, layer_combinations


def create_batch(batch_number):
    input_combinations, layer_combinations = get_combinations(create=False)
    folder_path = get_result_folder()

    if batch_number > 9:  # we only have 10 batches
        raise ValueError("The specified batch number does not exist")

    # DMF parameters
    P = DMF_parameters({})
    P['sigma'] = 0.02  # no noise

    # Simulation parameters
    t_sim = 3  # simulation time
    t_start = 0.5  # stimulation start time
    t_stop = 2.5  # stimulation stop time
    dt = P['dt']  # integration time step
    P['T'] = t_sim

    transient_end = int(0.1 / dt)  # end of initial transient

    # layer specific external input
    stim_start = int(t_start / dt)  # start of stimulation
    stim_end = int(t_stop / dt)  # end of stimulation
    sim_steps = int(t_sim / dt)  # number of simulation steps
    duration = sim_steps - transient_end

    stim_index = range(stim_start, stim_end)  # indices of stimulation
    input_template = np.zeros(sim_steps)  # input current
    # input current during stimulation (later just multiply with the specific input intensity)
    input_template[stim_index] = 1.0

    total_number_combinations = len(input_combinations)

    # divide the total number of combinations into 10 batches and save each batch separately
    batch_size = int(total_number_combinations / 10)
    Y = np.zeros(shape=(batch_size, duration, 8))

    batch_counter = 0
    index_batch_start = batch_size * batch_number
    for total_counter, (current, layer) in enumerate(zip(input_combinations[index_batch_start:, :],
                                                         layer_combinations[index_batch_start:, :])):
        if (total_counter + 1) % 50 == 0:
            print(f"Running simulation {total_counter + 1}/{batch_size}")
        U = np.zeros((sim_steps, 8))
        U[:, layer[0]] = input_template * current[0]
        U[:, layer[1]] = input_template * current[1]

        # Run the simulation
        I, _, _ = DMF_sim(U, P)
        Y[batch_counter, :, :] = I[transient_end:, :]
        batch_counter += 1
        # save if current batch is finished or all simulations are created
        if (total_counter + 1) % batch_size == 0 or total_counter == (input_combinations.shape[0] - 1):
            # Save the results and include the batch number in the file name with leading zeros
            # note that the baseline is not saved with the simulation to avoid large files
            np.save(os.path.join(folder_path, f'Y_{batch_number+1:02d}.npy'), Y)
            break  # we only want to do one batch at a time


if __name__ == '__main__':
    for batch in range(2, 9):  # file names 03 to 10
        print(f"Run batch for file Y_{batch + 1:02d}.npy")
        create_batch(batch)
