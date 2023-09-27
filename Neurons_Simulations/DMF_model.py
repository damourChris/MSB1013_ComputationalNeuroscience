import numpy as np
import copy

def DMF_sim(U, P):

    M = P['M']  # number of populations

    # --------------------------------------------------------------------------
    # Initial condtions:
    I = np.zeros((M))
    H = np.zeros((M))
    F = np.zeros((M))

    I_save = np.zeros((int(P['T'] / P['dt']), M))
    H_save = np.zeros((int(P['T'] / P['dt']), M))
    F_save = np.zeros((int(P['T'] / P['dt']), M))

    dt = P['dt']

    def f(h, a, b, d):
        # gain function
        h = np.float128(h)
        return (a * h - b) / (1 - np.exp(-d * (a * h - b)))

    for t in range(int(P['T'] / dt)):

        I += dt * (-I / P['tau_s'])
        I += dt * np.dot(P['W'], F)
        I += dt * (P['W_bg'] * P['nu_bg'])
        I += dt * U[t, :]
        I += np.sqrt(dt/P['tau_s']) * P['sigma'] * np.random.randn(M)
        H += dt * ((-H + P['R']*I) / P['tau_m'])
        F = f(H, a=P['a'], b=P['b'], d=P['d'])

        I_save[t, :] = copy.deepcopy(I)
        H_save[t, :] = copy.deepcopy(H)
        F_save[t, :] = copy.deepcopy(F)

    return I_save, H_save, F_save

def DMF_parameters(P):
    P['dt'] = 1e-4                          # integration step

    P['sigma'] = 0.002                      # noise amplitude
    P['tau_s'] = 0.5e-3                     # synaptic time constant
    P['tau_m'] = 10e-3                      # membrane time constant
    P['C_m']   = 250e-6                     # membrane capacitance
    P['R']     = P['tau_m'] / P['C_m']      # membrane resistance

    P['nu_bg'] = 8
    P['K_bg']  = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])

    M = 8                           # number of populations
    P['M'] = 8

    g = -4                          # inhibitory gain
    P['J_E'] = 87.8e-3              # excitatory synaptic weight
    P['J_I'] = P['J_E'] * g         # inhibitory synaptic weight

    P['W'] = np.tile([P['J_E'], P['J_I']], (M, int(M/2)))                                   

    P['P'] = np.array(                                                                      # connection probability
				     [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
				      [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
				      [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
				      [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
				      [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
				      [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
				      [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
				      [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
    # P['P'][0, 2] *= 2   # double the connection from L4E to L23E

    P['N'] = np.array([20683,  5834,   21915,  5479,   4850,   1065,   14395,  2948  ])     # number of neurons in each population
    P['K'] = np.log(1-P['P']) / np.log(1 - 1/(P['N'] * P['N'].T)) / P['N']                  # number of connections between populations

    P['W'] = P['W'] * P['K']                                                                # synaptic weight matri 

    P['W_bg'] = P['K_bg'] * P['J_E']                                                        # background synaptic weight

    # gain function parameters
    P['a'] = 48
    P['b'] = 981
    P['d'] = 8.9e-3

    return P


if __name__=="__main__":

    # run a single simulation

    import pylab as plt

    # parameters
    P = {}
    P = DMF_parameters(P)

    T = 1
    dt = 1e-4

    # input
    U = np.zeros((int(T/dt), P['M']))
    # U[int(0.2/dt):int(0.4/dt), 0] = 1
    # U[int(0.6/dt):int(0.8/dt), 2] = 1

    P['T'] = T

    # simulate
    I, H, F = DMF_sim(U, P)

    # plot
    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.title('input current [pA]')
    plt.plot(I)
    plt.xlim([0, 1000])
    plt.subplot(312)
    plt.title('membrane potential [mV]')
    plt.plot(H)
    plt.xlim([0, 1000])
    plt.subplot(313)
    plt.title('firing rate [Hz]')
    plt.plot(F)
    plt.xlim([0, 1000])
    plt.legend()
    plt.show()

    labels = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    colors = plt.tile(['b', 'r'], (int(P['M']/2)))

    plt.figure(figsize=(8, 8))
    # mean input current
    plt.subplot(131)
    plt.barh(np.arange(P['M']), np.mean(I, axis=0), color=colors)
    plt.yticks(np.arange(P['M']), labels)
    plt.xlabel('mean input current [pA]')
    plt.gca().invert_yaxis()
    # mean membrane potential
    plt.subplot(132)
    plt.barh(np.arange(P['M']), np.mean(H, axis=0), color=colors)
    plt.yticks(np.arange(P['M']), labels)
    plt.xlabel('mean membrane potential [mV]')
    plt.gca().invert_yaxis()
    # mean firing rate
    plt.subplot(133)
    plt.barh(np.arange(P['M']), np.mean(F, axis=0), color=colors)
    plt.yticks(np.arange(P['M']), labels)
    plt.xlabel('mean firing rate [Hz]')
    plt.gca().invert_yaxis()
    plt.show()
