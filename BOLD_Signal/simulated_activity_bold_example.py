import numpy as np
from matplotlib import pyplot as plt

from BOLD_Signal.helper import downsample_neural_activity, combine_inh_exc_only_exc, get_betas_from_neural_activity
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
    Y = I[int(1 / dt):, :] - np.mean(I[int(0.5 / dt):int(1 / dt), :],
                                     axis=0)  # deviation from baseline during stimulation (and remove initial transient)
    # Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) * 2 - 1  # normalize Y between -1 and 1
    Y = np.abs(Y * 50)

    B, X, bold_responses = get_betas_from_neural_activity(Y)

    downsampledY = downsample_neural_activity(combine_inh_exc_only_exc(Y))

    colors = plt.cm.Spectral(np.linspace(0, 1, 4))
    plt.figure(figsize=(10, 6))
    plt.subplot(411)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('Y')
    plt.plot(downsampledY)
    plt.xlim([0, len(downsampledY) - 1])
    plt.subplot(412)
    plt.title('B')
    plt.bar(['L23', 'L4', 'L5', 'L6'], B, color=colors)
    plt.subplot(413)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('X')
    plt.plot(X)
    plt.subplot(414)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('X*B')
    plt.plot(X * B)
    plt.xlim([0, len(X) - 1])
    plt.tight_layout()
    plt.show()
    # plt.savefig('fig/signal_betas_example.pdf', bbox_inches='tight', transparent=True, dpi=300)

    plt.figure()
    for layer in range(4):
        plt.subplot(4, 1, layer + 1)
        plt.plot(downsampledY[::int(2 / 0.001), layer], label="activity")
        plt.plot(bold_responses[:, layer], label="bold")
        plt.legend()
    # plt.savefig('fig/bold_neuronal_activity_example.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()
