import numpy as np
from matplotlib import pyplot as plt

from BOLD_Signal.helper import downsample_neural_activity, combine_inh_exc_only_exc, get_betas_from_neural_activity

if __name__ == '__main__':
    # create simple stimulus as neural activity
    t_sim = 50  # in s
    dt = 0.0001
    U = np.zeros(int(t_sim / dt))
    U[int(10 / dt):int(20 / dt)] = 1
    x = 0
    X = np.ones(int(t_sim / dt))
    for t in range(int(t_sim / dt)):
        x = x + dt * (U[t] - x)
        X[t] = x

    Y = np.zeros(shape=(int(t_sim / dt), 8))
    for layer in range(8):
        Y[:, layer] = X * layer / 10  # scale differently per layer
    # # depends on what the simulations look like, goal is to make it a bit bigger
    # Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) * 2 + 1  # normalize Y between -1 and 1

    B, X, bold_responses = get_betas_from_neural_activity(Y)

    downsampledY = downsample_neural_activity(combine_inh_exc_only_exc(Y))

    colors = plt.cm.Spectral(np.linspace(0, 1, 4))
    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('Y')
    plt.plot(downsampledY)
    plt.xlim([0, len(downsampledY) - 1])
    plt.subplot(312)
    plt.title('B')
    plt.bar(['L23', 'L4', 'L5', 'L6'], B, color=colors)
    plt.subplot(313)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('X*B')
    plt.plot(X * B)
    plt.xlim([0, len(X) - 1])
    plt.tight_layout()
    plt.savefig('fig/signal_betas_example.pdf', bbox_inches='tight', transparent=True, dpi=300)

    plt.figure()
    for layer in range(4):
        plt.subplot(4, 1, layer + 1)
        plt.plot(downsampledY[::int(2/0.001), layer], label="activity")
        plt.plot(bold_responses[:, layer], label="bold")
        plt.legend()
    plt.savefig('fig/bold_neuronal_activity_example.pdf', bbox_inches='tight', transparent=True, dpi=300)
