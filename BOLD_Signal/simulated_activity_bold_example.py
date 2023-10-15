import numpy as np

from BOLD_Signal.helper import get_betas_from_neural_activity, \
    plot_neural_activity_and_betas, plot_neural_activity_and_bold
from Neurons_Simulations.DMF_model import DMF_parameters, DMF_sim

if __name__ == '__main__':
    # DMF parameters
    P = {}
    P = DMF_parameters(P)
    P['sigma'] = 0.02  # no noise

    T = 30  # simulation time
    dt = 1e-4  # integration step
    P['T'] = T

    # layer specific external input
    stim_start = int(10 / dt)  # start of stimulation
    stim_end = int(20 / dt)  # end of stimulation
    NU = [0, 0, 200, 150, 0, 0, 0, 0]  # input to model populations; this will be our set of parameters to be inferred.
    U = np.zeros((int(T / dt), P['M']))
    U[stim_start:stim_end, 0] = NU[0]  # L23E
    U[stim_start:stim_end, 1] = NU[1]  # L23I
    U[stim_start:stim_end, 2] = NU[2]  # L4E
    U[stim_start:stim_end, 3] = NU[3]  # L4I
    U[stim_start:stim_end, 4] = NU[4]  # L5E
    U[stim_start:stim_end, 5] = NU[5]  # L5I
    U[stim_start:stim_end, 6] = NU[6]  # L6E
    U[stim_start:stim_end, 7] = NU[7]  # L6I

    ### SIMULATION ###
    # simulate DMF (i.e. neural activity)
    I, H, F = DMF_sim(U, P)  # I - input current, H - membrane potential, F - firing rate

    # deviation from baseline during stimulation (and remove initial transient)
    Y = I[int(1 / dt):, :] - np.mean(I[int(0.5 / dt):int(1 / dt), :], axis=0)

    B, X, bold_responses, neural_activity = get_betas_from_neural_activity(Y)

    plot_neural_activity_and_betas(neural_activity, B, X)

    plot_neural_activity_and_bold(neural_activity, bold_responses)
