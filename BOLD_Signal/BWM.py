import numpy as np

# TODO: check whether default values are appropriate
def balloon_windkessel(f, dt=0.001,
                       tau_mtt=2.0, alpha=0.32,
                       E_0=0.4, V_0=4,
                       tau_vs=4, epsilon=0.0463, rho_0=0.191, nu_0=126.3, TE=0.028):
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
    k1 = 4.3 * nu_0 * E_0 * TE
    k2 = epsilon * rho_0 * E_0 * TE
    k3 = 1 - epsilon
    n_time_points = len(f)
    v = np.zeros(n_time_points)
    q = np.zeros(n_time_points)
    bold = np.zeros(n_time_points)

    plt.figure()
    plt.plot(f)
    plt.show()

    f_out = 0
    curr_v = 0
    curr_q = 0

    for t in range(0, n_time_points):
        E_f = 1 - (1 - E_0) ** (1 / f[t])

        dv = dt * ((f[t] - f_out) / tau_mtt)  # blood volume

        curr_v = curr_v + dv
        f_out = curr_v ** (1 / alpha) + tau_vs * dv

        dq = dt * (f[t] * (E_f / E_0) - f_out * (curr_q / curr_v) / tau_mtt)  # deoxyhemoglobin

        v[t] = curr_v
        curr_q = curr_q + dq
        q[t] = curr_q

        bold[t] = V_0 * (k1 * (1.0 - q[t]) + k2 * (1.0 - q[t] / v[t]) + k3 * (1.0 - v[t]))

    return bold, v, q


# Example usage:
import matplotlib.pyplot as plt

# Create a simple block design neural activity
t_sim = 40 # in s
dt = 0.001
input = np.zeros(int(t_sim/dt))
input[int(10/dt):int(15/dt)] = 1
X = np.ones(int(t_sim/dt))
for t in range(int(t_sim/dt)):
    X[t] = X[t-1] + dt * (input[t] - X[t-1] + 1)

# Get the BOLD response
bold, v, q = balloon_windkessel(X, dt=0.001)
print(bold)

# Plot the results
plt.figure(figsize=(10, 10))
plt.subplot(6,1,1)
plt.plot(input, label="stimulus")
plt.subplot(6,1,2)
plt.plot(X, label='Neural Activity')
plt.subplot(6,1,3)
plt.plot(bold, label='BOLD Response')
plt.subplot(6,1,4)
plt.plot(v, label='v')
plt.subplot(6,1,5)
plt.plot(q, label='q')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
