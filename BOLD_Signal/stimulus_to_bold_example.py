import numpy as np
from matplotlib import pyplot as plt

from BOLD_Signal.helper import downsample_neural_activity, combine_inh_exc_only_exc, get_betas_from_neural_activity
from BOLD_Signal.plot_utils import get_plt_settings, get_plt_size

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

    B, X, bold_responses, neural_activity_normalised = get_betas_from_neural_activity(
        Y,
        stim_start=10,
        stim_end=20,
        neural_activity_sampling_rate=dt
    )

    # settings for matplotlib
    plt.rcParams.update(get_plt_settings())
    width, height = get_plt_size(1.0)

    colors = plt.cm.Spectral(np.linspace(0, 1, 4))
    plt.figure(figsize=(width, height))
    plt.suptitle("Estimated parameters of the linear regression")
    plt.subplot(311)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('Neural signal')
    plt.plot(np.arange(0, t_sim, 0.001), neural_activity_normalised)
    plt.ylabel("Strength of activity")
    plt.xlabel("t [sec]")
    plt.subplot(312)
    plt.title('Feature weights (betas)')
    plt.bar(['L23', 'L4', 'L5', 'L6'], B, color=colors)
    plt.subplot(313)
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Spectral(np.linspace(0, 1, 4))))
    plt.title('Predicted input')
    plt.plot(np.arange(0, t_sim, 2), X * B)
    plt.ylabel("Strength of activity")
    plt.xlim([0, len(X)*2])
    plt.xlabel("t [sec]")
    plt.tight_layout()
    plt.savefig('fig/signal_betas_example.pdf', bbox_inches='tight', transparent=True, dpi=300)

    fig = plt.figure()
    plt.figure(figsize=(width, height*1.5))
    plt.suptitle("Predicted BOLD response of the Balloon-Windkessel model")
    for layer in range(4):
        ax1 = plt.subplot(4, 1, layer + 1)
        plt.title(f"Layer {layer + 1}")
        t = np.arange(0, bold_responses.shape[0]*2 - 1, 2)

        neural_activity_line,  = ax1.plot(t, neural_activity_normalised[::int(2/0.001), layer], label="Neural activity",
                                          color='C0')
        ax1.set_yticks(np.arange(0, np.max(neural_activity_normalised)*1.2, 0.25))
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.set_ylabel("Strength of activity")
        ax1_ylim_min = -0.02
        ax1_ylim_max = 1.17

        ax1.set_ylim([ax1_ylim_min, ax1_ylim_max])

        ax1_ylim_range = ax1_ylim_max - ax1_ylim_min

        ax2 = ax1.twinx()
        bold_line,  = ax2.plot(t, bold_responses[:, layer], label="BOLD signal", color='C1')
        ax2.set_yticks(np.arange(0, np.max(bold_responses)+2, 4))
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylabel("BOLD signal [\%]")
        ax2.set_ylim([ax1_ylim_min/ax1_ylim_range*(np.max(bold_responses)+2),
                      ax1_ylim_max/ax1_ylim_range*(np.max(bold_responses)+2)])

        ax1.set_xlim([0, bold_responses.shape[0]*2])
        ax1.set_xlabel("t [sec]")
        ax1.legend([neural_activity_line, bold_line], ['Neural activity', 'BOLD signal'])
    plt.tight_layout(h_pad=1)
    plt.savefig('fig/bold_neuronal_activity_example.pdf', bbox_inches='tight', transparent=True, dpi=300)
