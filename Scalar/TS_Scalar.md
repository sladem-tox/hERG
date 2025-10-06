#Thompson Sampling for Scalar Predictive Models

1. Mote Carlo Dropout Sampling

During prediction, dropout layers are kept active so that the model performs multiple stochasitc forward passes. Here set to 20.

Each pass uses a different set of network weights due to the MC dropout. This produces a set of predictions for each molecule in the pool.

This means that each molecule has a predictive distribution rather than a single prediciton.

2. Uncertainty Estimation

The standard deviation of these predictions across the MC samples is computed for each molecule.

Importantly, this SD acts as a proxy for the model's epistemic uncertainty. High SD and the model is less certain about the predicted value.

3. Query Selection

The molecule with the highest SD is selected as the next "query point".

4. Dataset Update

The selected molecule is then removed from the pool and added to the training set. The model is retrained on the expanded training set.

This active learning cycle continues for q_num number of queries (user defined), gradually improving model performance.
The number of samples required for training is thus minimised for maximum information extraction.

