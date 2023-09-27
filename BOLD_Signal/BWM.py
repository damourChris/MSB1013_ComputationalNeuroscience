import numpy as np


# TODO: check wether default values are appropriate
def balloon_windkessel(neural_activity, dt=0.1,
                       tau=0.98, alpha=0.32,
                       E_0=0.34, V_0=0.02,
                       k1=7, k2=2, k3=2,
                       tau_s=0.8, tau_f=0.4):
    """
    Simulates the BOLD response using the Balloon-Windkessel model.
    
    :param neural_activity: array-like, the neural activity over time
    :param dt: float, the time step
    :param tau: float, the hemodynamic transit time
    :param alpha: float, the Grubb's exponent
    :param E_0: float, the resting oxygen extraction fraction
    :param V_0: float, the resting blood volume fraction
    :param k1, k2, k3: floats, the BOLD signal coefficients
    :param tau_s: float, the time constant for the signal decay
    :param tau_f: float, the time constant for the signal rise
    
    :return: array-like, the BOLD response over time
    """
    n_time_points = len(neural_activity)
    s = np.zeros(n_time_points)
    f = np.zeros(n_time_points)
    v = np.ones(n_time_points) * V_0
    q = np.ones(n_time_points) * V_0 * E_0
    bold = np.zeros(n_time_points)

    for t in range(1, n_time_points):
        ds = dt * (-s[t - 1] / tau_s + neural_activity[t - 1])  # neural activity
        df = dt * (s[t - 1] - f[t - 1]) / tau_f  # vasodilatory signal
        dv = dt * ((f[t - 1] - v[t - 1] ** (1 / alpha)) / tau)  # blood volume
        # TODO: we get a RuntimeWarning (divide by zero) here, looks like v[t-1] is the reason
        dq = dt * ((f[t - 1] * (1 - (1 - E_0) ** (1 / f[t - 1])) - (v[t - 1] ** (1 - alpha) * q[t - 1]) / v[t - 1]) / tau)  # deoxyhemoglobin

        s[t] = s[t - 1] + ds
        f[t] = f[t - 1] + df
        v[t] = v[t - 1] + dv
        q[t] = q[t - 1] + dq

        bold[t] = v[t] * (k1 + k2) - (k2 / k3) * q[t]

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
plt.plot(bold_response, label='BOLD Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
