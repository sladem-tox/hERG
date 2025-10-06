# hERG Binding Models with Active Learning

1. Binary

These models are binary Thompson sampling active learning models with a threshold set at some biologically relevant levels such as 1 or 10 micromolar.

Uncertainty is determined by contrasting predictions with 0.5 thresold and the smallest difference is assumed most uncertain.

2. Scalar

These models attempt to predict pIC50 values directly. A Thompson sampling active learning approach is used.
Uncertainty is determined via application of MC dropout on the neural network to generate a prediction sample for each molecule.

## Setup

Use the environment.yaml file to create a conda environment with all dependencies.

E.g.

`conda env create -n new_env_name -f environment.yaml`

Then the scripts are set up to take 3 arguments: training_file.csv, test_file.csv, and --q_num [number of queries]

At this stage just set up for 2048 bit ECFP's for simplicity.



