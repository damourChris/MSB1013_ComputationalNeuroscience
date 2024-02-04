# General Linear Model

The General Linear Model (GLM) is widely used in fMRI analysis, combining knowledge about
the experimental design, i.e. the timing of the stimulus and the BOLD morphology. The general
idea is to find parameters linking the obtained BOLD responses to an idealized BOLD response,
describing the contribution of each of the layers. Because the BOLD signal is known to follow
a consistent shape, it can be modelled using the **hemodynamic response function** (HRF) [25].
This is used as our design matrix. By combining it via convolution with our condition array
indicating the time interval of the stimulus and normalising the result between 0 and 1, we
obtain an idealized BOLD response for each layer.

GLM is then used to obtain a set of parameters \\(β\\)s that describe the relation between the
simulation of the BOLD response Y given the neural active and the created model X, which only
contains information about the timing of the stimulus. Therefore, each coefficient \\(β\\) quantifies
the amount of activity seen in each layer. In this project’s scope, each layer is considered a voxel.

The GLM is expressed as

\\[Y = Xβ + e.\\]

with \\(Y ∈ ℝ^{T×4}\\) of the simulated BOLD responses, \\(X ∈ ℝ^{T×4}\\) of the model, the parameters
\\(β ∈ ℝ^{4×4}\\) and \\(e ∈ ℝ^{T×N}\\) being the residual term. This is mathematically identical to multiple
regression analysis and can be solved using Least Squares to find the solution with the minimized
sum of squared error terms [26]:
\\[ \hat{β} = (X′X)^{−1}X′y \\]
Thus, we end with parameters describing the amplitude of each voxel’s fMRI. To mirror a realistic
fMRI sample rate, the BOLD signal and the idealized fMRI are both down-sampled to 0.5 Hz
before solving the GLM.