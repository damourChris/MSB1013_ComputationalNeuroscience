import os
import numpy as np
from matplotlib import pyplot as plt

# Before starting, change the cwd if needed by uncommenting and inserting the right path.
#os.chdir("/home/coder/projects/Computational Neuroscience")

# ---------------------------------------------------------------------------------------------
# Check how X looks like:

data_X = np.load("X.npy")

params = data_X[:, -3:]

print("The length of X is: ", len(data_X[0]))
print("The shape of X is: ", data_X.shape)
print("The first row/simulation results are: ", "\n", data_X[0])

# ---------------------------------------------------------------------------------------------
# Incorporation of noise (general):

# First create noise matrix that has the same size as (data_)X matrix.
# random.randn() returns "random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1."
# Use a seed when you want the same outcome everytime!
rows_X, columns_X = data_X.shape

# Add noise to summary statistics (in Lorenz example, len of X is 24, where the last 3 columns contain the parameters)
# so the first 21 columns contain summary statistics.
noise_matrix = np.random.randn(rows_X, 21)

# Don't add noise to the parameters.
matrix_with_zeros = np.zeros(shape = (rows_X, 3))

# Create noise_matrix with the right size so it can be added up to the (data_)X matrix.
noise_matrix = np.concatenate((noise_matrix, matrix_with_zeros), axis = 1)

# Add noise to (data_)X:
summ_stats_with_noise = data_X + noise_matrix
print("The first row of (data_)X + noise values: ", "\n", summ_stats_with_noise[0])

# ---
# Incorporation of noise for SBI (can't use this yet as X has a different shape in the Lorenz example than in our case).
# So for testing purposes (influence of noise on summary statistics for Lorenz example), use the code above.

# # In our case, if we want to add noise only to the beta values/summary statistics then the following code can be used:
# rows_X, columns_X = data_X.shape

# # We want the same rows but only the last 4 colums of the (data_)X matrix contains the beta values.
# noise_matrix = np.random.randn(rows_X, 4)

# matrix_with_zeros = np.zeros(shape = (rows_X, 8))

# # Create noise_matrix with the right size so it can be added up to the (data_)X matrix.
# noise_matrix = np.concatenate((matrix_with_zeros, noise_matrix), axis = 1)

# # Add noise to (data_)X:
# summ_stats_with_noise = data_X + noise_matrix
# print("The first row of (data_)X + noise values: ", "\n", summ_stats_with_noise[0])









