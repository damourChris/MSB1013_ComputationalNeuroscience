# Testing 
To test the generated posterior models, a testing method was established to evaluate statistics
based on a testing matrix. This testing matrix is split similarly to the training data, where
9
the beta values and parameters are separated into the observation and the target parameters.
The aim of the testing was two-fold: To assess whether it could infer which populations were
active given the summary statistics generated via the BOLD model and whether the model
could actually predict the level of activation given to a specific population. The hypothesis was
that the model was more likely to predict whether a population was active or not rather than
being able to establish the level of activation. When testing a prediction, the posterior is given
an observation via set_default_x(), and then sampled according to the testing parameter
num_sample. This produces an array of predictions for a given observation. To establish the
prediction of the activity of a given population, the value of the 50th percentile is calculated.
A prediction array can then be built on the value predicted for each population. To assess the
quality of the inference, multiple metrics were created.