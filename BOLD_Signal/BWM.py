import numpy as np

# TODO: check whether default values are appropriate
def balloon_windkessel(neural_activity, dt=0.1,
                       tau=0.98, alpha=0.32,
                       E_0=0.8, V_0=0.02,
                       tau_s=0.8, tau_f=0.4, epsilon=0.5):
    # alpha is 0.2 in Friston 2000
    # E_0 is 0.8 in Friston 2000 but also 0.34 in neurolib
    """
    Simulates the BOLD response using the Balloon-Windkessel model.
    
    :param neural_activity: array-like, the neural activity over time
    :param dt: float, the time step
    :param tau: float, the hemodynamic transit time
    :param alpha: float, the Grubb's exponent
    :param E_0: float, the resting oxygen extraction fraction, also called rho
    :param V_0: float, the resting blood volume fraction
    :param tau_s: float, the time constant for the signal decay
    :param tau_f: float, the time constant for the signal rise
    :param epsilon: float, unknown parameter from Friston2000
    
    :return: array-like, the BOLD response over time
    """
    k1 = 7 * E_0
    k2 = 2
    k3 = 2 * E_0 - 0.2
    n_time_points = len(neural_activity)
    s = np.zeros(n_time_points)
    f = np.ones(n_time_points)
    v = np.ones(n_time_points)  # * V_0
    q = np.ones(n_time_points)  # * V_0 * E_0
    bold = np.zeros(n_time_points)

    for t in range(1, n_time_points):
        # equations same as in Friston 2000
        f_out = v[t - 1] ** (1 / alpha)
        ds = dt * (-s[t - 1] / tau_s + epsilon * neural_activity[t - 1] - (f[t-1] - 1) / tau_f)  # neural activity
        df = dt * s[t - 1]  # vasodilatory signal
        dv = dt * ((f[t - 1] - f_out) / tau)  # blood volume
        # TODO: we get a RuntimeWarning (divide by zero) here, looks like v[t-1] is the reason
        dq = dt * (f[t - 1] * ((1 - (1 - E_0) ** (1 / f[t - 1])) / E_0) - (f_out * q[t - 1]) / v[t - 1]) / tau  # deoxyhemoglobin

        s[t] = s[t - 1] + ds
        f[t] = max(f[t - 1] + df, 1e-120)
        v[t] = v[t - 1] + dv
        q[t] = q[t - 1] + dq

        bold[t] = V_0 * (k1 * (1.0 - q[t]) + k2 * (1.0 - q[t] / v[t]) + k3 * (1.0 - v[t]))

    return bold


# Example usage:
import matplotlib.pyplot as plt

# Create a simple block design neural activity
neural_activity = np.zeros(1000)
neural_activity[100:200] = 1
neural_activity[400:500] = 1
neural_activity[700:800] = 1

# Get the BOLD response
bold_response = balloon_windkessel(neural_activity)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(neural_activity, label='Neural Activity')
plt.plot(bold_response*100, label='BOLD Response (scaled by 100)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
