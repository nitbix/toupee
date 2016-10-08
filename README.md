# toupee
"The ugly thing on top that covers up what's missing"
A library for Deep Learning experiments that also includes Ensembles, based on
Keras.

Usage:

Experiments are described in a common YAML format, and each network structure is
in serialised Keras format.

Supports saving results to MongoDB for analysis later on.

In bin/ you will find 2 files:

 * *mlp.py*: takes an experiment description and runs it as a single network.
   Ignores all ensemble directives.

 * *ensemble.py*: takes an experiment description and runs it as an ensemble.

In examples/ there are a few ready-cooked models that you can look at.
