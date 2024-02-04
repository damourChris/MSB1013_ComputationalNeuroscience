# Metrics
## Test specific metrics

- The first metric was a binary prediction of whether a layer is predicted to be active or
not. The testing parameter activation_threshold establishes at what value a population
input value should be to be considered active. A binary prediction array is then built to
highlight the populations the prediction deemed active. This is done on the true values
and then compared to establish if the prediction was able to infer which populations were
correctly active.
- The second metric is the percentage of samples that give a value that lies in a confidence
range. As such, two parameters were established: threshold and tolerance. The
threshold is a percentage of the true value and the tolerance is an absolute value. The range is then given as:
\\[
truevalue(1 − threshold) − tolerance ≤ predictedvalue ≤ truevalue(1 + threshold) + tolerance
\\]

An array is then constructed to keep track of the range of values predicted by the sampling.
This allows for a simpler interpretation of the shape of the distribution. Indeed, it can be
understood as the confidence of the prediction in the actual value.

## Model specific metrics
To evaluate the performance of the model on a particular testing data set, the balanced accuracy
was calculated. This was first done on each population, and then on combinations of activation.
The rationale behind this was that a prediction might be able to detect one population but not
the other one in the case where two populations were activated. As such, a confusion matrix
alongside the balanced accuracy using the binary prediction array was computed per population.
Furthermore, the choice of balanced accuracy was motivated by the fact that the data set was
heavily unbalanced where for a given population it was inactivated around 80% of the time. The
balanced accuracy was calculated as follows:
\\[\text{Balanced}_\text{accuracy} = \frac{1}{2}(\frac{TP}{TP + FP}+\frac{TN}{TN + FN}) \\]

where \\(TP\\) = True positives, \\(FP\\) = False positives, \\(TN\\) = True negatives and \\(FN\\) = False
negatives
A confusion matrix was also calculated for each combination of populations to investigate the
prediction accuracy further. That is, given a prediction, was the model able to predict the
activation state of two layers accurately?

## Data set specific metrics
Finally, the overall accuracy was evaluated per model by averaging the balanced accuracy evaluated
per population.