# Welcome to Toupee
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

## Quick-start

* Install keras
* Clone this repo
* In examples/ there are a few working examples of experiments:
  * Download the needed dataset [here](http://dropbox.com/toupee-datasets), and
    save it to the correct location (or change the location in the example)
* Run bin/mlp.py for single network experiments, bin/ensemble.py for ensemble
  experiments

### Datasets

Datasets are saved in the `.npz` format, with three files in a directory:
* `train.npz`: the training set
* `valid.npz`: the validation set
* `test.npz`: the test set
Each of these files is a serialised dictionary `{x: numpy.array, y: numpy.array}`
where `x` is the input data and `y` is the expected classification output.

### Experiment files

### Model files
