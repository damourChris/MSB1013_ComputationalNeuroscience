# Neural Net Inference

A neural network was used to relate the summary statistics contained in the beta values (output
of the previous GLM model) to the input injected into the various neuronal populations in the
four cortical layers. The simulation-based inference (SBI) toolbox, a PyTorch package, was used
for this purpose.
We adapted the code available in the following [GitHub repository from Kris Evers](https://github.com/krisevers/lorenz_sbi/tree/main).

In short, an "X matrix" in the form of a
2D NumPy array is created containing the original 8 input currents parameters and their 4 corresponding
beta values. Every array corresponds to one simulation. In total, 34560 simulations
were performed, implying that the size of X is 34560 x 12. As the generation of the data matrix
was done in a linear manner, to ensure that the test data had an even distribution of combinations,
the matrix was randomly shuffled and split into a train and test set where 10% (3456
simulations) are assigned as test data. 

In the process of training the neural network, the neural network has an initial density estimator attached to the model. The training set was further
divided into the parameters and beta values by separating the last 4 columns containing the
beta values from the first eight columns that contain the parameters. Then, the model can be
trained using the Sequential Neural Posterior Estimation (SNPE) algorithm. Simulation data is
appended to the model in order to establish training data for the model. [27]. This was done with
the help of a Masked Autoregressive Flow (MAF) neural density estimator. 

After training the model, this model was saved as an intermediary result to avoid the need to retrain the network
for future implementations.
It is worth noting that in the SBI toolbox, 2 other algorithms are given, namely SNLE (for
likelihood estimation) and SNRE (for likelihood-ratio estimation). However, these two methods
require a prior distribution to be defined prior to training. Due to the nature of the simulation
data, this approach was not pursued.
The process of shuffling, splitting and training the model was performed 10 times to avoid the
scenario where a bad training and/or test set is acquired by chance. That is, 10 different posterior
models were trained (posterior_n, where n = 1 to 10) with their corresponding training and test
set, numbered from 1 to 10.