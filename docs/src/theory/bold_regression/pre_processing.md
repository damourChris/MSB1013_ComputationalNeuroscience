# Neural Signal Pre-processing

The BOLD signal is best correlated to changes in local-field potential (LFP) [23], which can be
estimated by summing the values of the currents from the excitatory and inhibitory populations.
[24] However, the model used in this project is tuned for taking inputs from a single excitatory
population.[22] 

For our simulations, the neural signal involved in the neurovascular coupling is
the sum of the absolute values of inhibitory and excitatory populations per layer. We further
5
down-sample this current to a sampling rate of 1000 Hz, a time resolution sufficient to estimate
the BOLD response. We also scale the amplitude of the signal, as the balloon model is tuned
for neural activity described in terms of the proportion of activity with values between 0 and 1.
Since we are mainly interested in the relative proportions of the activity in the different layers,
the maximum value of all layers in one simulation is mapped to the maximal activity 1. 

The baseline is removed by using the mean values during the absence of a stimulus. Furthermore,
absolute values are used because only the strength of the current is relevant. We refer to this as
the neural activity input for the BOLD model. The model will be applied to each layer separately
and treated as independent, as the balloon model does not account for (cortical) depth.