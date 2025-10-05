## Active Learning / Model Experiments for DropoutNet

*0. You can time the scripts with $time python script.py*


*1. Baseline*

Current network: 2 hidden layers (512 → 256), leaky_relu, p_dropout=0.3

Train as is, track F1, Accuracy, Precision, Recall.

*2. Depth and Width*

Experiment 2a: Add one more hidden layer (512 → 256 → 128).

Experiment 2b: Increase width of first hidden layer (1024 → 512).

Experiment 2c: Deep + wide (1024 → 512 → 256 → 128).

*3. Activation Functions*

Experiment 3a: Replace leaky_relu with ReLU.

Experiment 3b: Replace with ELU.

Experiment 3c: Replace with GELU.

Keep dropout same; compare metrics.

*4. Dropout Variations*

Experiment 4a: Change dropout to p=0.2 first layer, p=0.5 second.

Experiment 4b: Remove dropout entirely to see effect on overfitting.

Experiment 4c: Use MC Dropout at different layers (keep dropout in inference).

*5. Batch Normalization*

Experiment 5a: Add nn.BatchNorm1d after each hidden layer.

Experiment 5b: BatchNorm + dropout together.

*6. Residual / Skip Connections*

Experiment 6a: Simple residual connection between hidden layers.

Experiment 6b: Residual + batch norm.

*7. Optimizer / Training Variations*

Experiment 7a: Try AdamW optimizer instead of Adam.

Experiment 7b: Add L2 weight decay (1e-5–1e-4).

Experiment 7c: Use OneCycleLR scheduler for dynamic learning rate.

Experiment 7d: Early stopping based on validation F1.

*8. Input / Feature Variations*

Experiment 8a: Try standardizing / normalizing input features.

Experiment 8b: Add additional molecular descriptors if available.

Experiment 8c: Try PCA-reduced features to lower dimensionality.

*9. Advanced Architectures*

Experiment 9a: 1D CNN over fingerprint input.

Experiment 9b: Graph Neural Network (GCN, GAT) using molecular graph input.

Experiment 9c: Ensemble of 3–5 DropoutNet models (average predictions).

*10. Evaluation Tracking*

# Always save:

Test set metrics: F1, Accuracy, Precision, Recall

Learning curve: F1 over active learning iterations

Predictions CSV + model weights

Optionally, use Rich to print metrics in color during loop.

💡 Tip:

Make a new Python file for each experiment, e.g., exp_2a_deeper_net.py, exp_3b_ELU.py.

Keep a results log: experiment_results.csv with columns: Experiment, F1, Accuracy, Precision, Recall, Notes.
