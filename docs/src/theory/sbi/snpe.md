For Bayesian Inference, the goal is to infer simulation settings via deep learning. In the case of
Sequential Neural Posterior Estimation (SNPE), the process consists of generating a posterior
distribution of the system by training the neural network with a set of simulation parameters
and associated results. In practice, given simulation results, one can sample this distribution
to estimate the original simulation parameters that were used to generate these results. This
distribution is essentially a density estimator that indicates how often the system would generate
the data given a set of parameters. In this case, it was achieved with a Masked Autoregressive
Flow (MAF) neural density estimator.