import numpy as np


def get_ballon_windkessel_params():
    # TODO: check whether descriptions are correct
    """
    E_0: float, the resting oxygen extraction fraction, also called rho
    V_0: float, the resting blood volume fraction
    tau_vs: float, the time constant for the signal decay
    tau_mtt: float, the hemodynamic transit time
    alpha: float, the Grubb's exponent
    """
    return {
        'c1': 0.6,
        'c2': 1.5,
        'c3': 0.6,
        'tau_mtt': 2,
        'tau_vs': 4,
        'alpha': 0.32,
        'E_0': 0.4,
        'V_0': 4,
        'eps': 0.0463,
        'rho_0': .191,
        'nu_0': 126.3,
        'TE': 0.028
    }


def balloon_windkessel(neural_activity, stim_start, dt=0.001):
    """
    Simulates the BOLD response using the Balloon-Windkessel model.
    
    :param neural_activity: array-like, the neural activity over time
    :param stim_start: start of the stimulation in s, used to remove bold baseline
    :param dt: float, the time step
    
    :return: array-like, the BOLD response over time
    """
    theta = get_ballon_windkessel_params()
    k1 = 4.3 * theta['nu_0'] * theta['E_0'] * theta['TE']
    k2 = theta['eps'] * theta['rho_0'] * theta['E_0'] * theta['TE']
    k3 = 1 - theta['eps']
    n_time_points = len(neural_activity)

    v = np.zeros(n_time_points)
    q = np.zeros(n_time_points)
    bold = np.zeros(n_time_points)

    f = np.zeros(n_time_points)

    theta = {'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
             'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 4, 'eps': 0.0463, 'rho_0': .191,
             'nu_0': 126.3, 'TE': 0.028}

    xinflow = 0
    xvaso = 0
    yinflow = 0
    yvaso = 0

    # neurovascular coupling, also performs low pass
    for t in range(n_time_points):
        xinflow = np.exp(xinflow)
        yvaso = yvaso + dt * (neural_activity[t] - theta['c1'] * xvaso)  # vasoactive signal
        df_a = theta['c2'] * xvaso - theta['c3'] * (xinflow - 1)  # inflow
        yinflow = yinflow + dt * (df_a / xinflow)
        xvaso = yvaso
        xinflow = yinflow
        f[t] = np.exp(yinflow)
    # f = np.nan_to_num(f) # TODO: we should not get nan values

    f_out = 0
    curr_v = 1
    curr_q = 1

    for t in range(n_time_points):
        E_f = 1 - (1 - theta['E_0']) ** (1 / f[t])  # oxygen extraction fraction

        dv = dt * ((f[t] - f_out) / theta['tau_mtt'])  # blood volume

        dq = dt * ((f[t] * (E_f / theta['E_0']) - f_out * (curr_q / curr_v)) / theta['tau_mtt'])  # deoxyhemoglobin

        f_out = curr_v ** (1 / theta['alpha']) + theta['tau_vs'] * dv

        curr_v = curr_v + dv
        v[t] = curr_v
        curr_q = curr_q + dq
        q[t] = curr_q

        bold[t] = theta['V_0'] * (k1 * (1.0 - q[t]) + k2 * (1.0 - q[t] / v[t]) + k3 * (1.0 - v[t]))

    bold = bold - bold[int(stim_start/dt)-1]
    bold[:int(stim_start/dt)] = 0
    return bold, f, v, q


if __name__ == '__main__':
    # Example usage:
    import matplotlib.pyplot as plt

    # Create a simple block design neural activity
    t_sim = 50  # in s
    dt = 0.001
    U = np.zeros(int(t_sim / dt))
    U[int(10 / dt):int(15 / dt)] = 1
    x = 0
    X = np.ones(int(t_sim / dt))
    for t in range(int(t_sim / dt)):
        x = x + dt * (U[t] - x)
        X[t] = x

    # Get the BOLD response
    bold, f, v, q = balloon_windkessel(X, stim_start=10, dt=0.001)

    # Plot the results
    plt.figure(figsize=(5, 10))
    plt.subplot(6, 1, 1)
    plt.plot(U[6000:], lw=3)
    plt.title('Stimulus')
    plt.subplot(6, 1, 2)
    plt.plot(f[6000:])
    plt.title('Cerebral Blood Flow')
    plt.subplot(6, 1, 3)
    plt.plot(v[6000:], lw=3)
    plt.title('Cerebral Blood Volume')
    plt.subplot(6, 1, 4)
    plt.plot(q[6000:], lw=3)
    plt.title('Deoxyhemoglobin Content')
    plt.subplot(6, 1, 5)
    plt.plot(bold[6000:], lw=3)
    plt.title('BOLD Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
