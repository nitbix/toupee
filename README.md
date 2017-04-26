# Welcome to Toupee
"The ugly thing on top that covers up what's missing"
A library for Deep Learning ensembles, with a tooolkit for running experiments,
based on Keras.

Usage:

Experiments are described in a common YAML format, and each network structure is
in serialised Keras format.

Supports saving results to MongoDB for analysis later on.

In bin/ you will find 3 files:

 * *mlp.py*: takes an experiment description and runs it as a single network.
   Ignores all ensemble directives.

 * *ensemble.py*: takes an experiment description and runs it as an ensemble.

 * *distilled_ensemble.py*: takes an experiment description and runs it as an
   ensemble, and then distils the ensemble into a single network.

In examples/ there are a few ready-cooked models that you can look at.

## Quick-start

* Install keras
* Clone this repo
* In examples/ there are a few working examples of experiments:
  * Download the needed dataset [here](https://www.dropbox.com/sh/zqxqyx9g9rjhiuo/AACLpzyG-YC2BAKjV4yEaOmwa?dl=0),
    and save it to the correct location (or change the location in the example)
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

THIS SECTION IS INCOMPLETE

This is the file given as an argument to `mlp.py`, `ensemble.py` or
`distilled_ensemble.py`. It is a yaml description of the experiment.
Here is an example experiment file:

```
---
## MLP Parameters ##
dataset: /local/mnist_th/
pickled: false
model_file: mnist.model
optimizer:
  class_name: WAME
  config:
    lr: 0.001
    decay: 1e-4
n_epochs: 100 #max number of training epochs
batch_size: 128
cost_function: categorical_crossentropy
shuffle_dataset: true

## Ensemble Parameters ##
resample_size: 60000
method: !AdaBoostM1 { }
ensemble_size: 10
```

The parameters are as follows:

(network parameters)
 - `dataset`: the location of the dataset. If in "pickle" format, this is a
   file; if in "npz" format, this is a directory.
 - `pickled`: true if the dataset is in "pickle" format, false if "npz". Default
   is false.
 - `model_file`: the location of the serialised Keras model description.
 - `optimizer`: the SGD optimization method. See separate section for
   description.
 - `n_epochs`: the number of training epochs
 - `batch_size`

(optimizer subparameters)
### Model files
These are standard Keras models, serialised to yaml. Effectively, this is the
verbatim output of a model's `to_yaml()`.
