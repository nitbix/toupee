# Welcome to Toupee
![Toupee Logo](/logo.png)

"The ugly thing on top that covers up what's missing"

A library for Deep Learning ensembles, with a tooolkit for running experiments,
based on Keras.

Usage:

Experiments are described in a common YAML format, and each network structure is
in serialised Keras format.

Supports saving results to MongoDB for analysis later on.

In bin/ you will find two files:

 * *base_model.py*: takes an experiment description and runs it as a single network.
   Ignores all ensemble directives.

 * *ensemble.py*: takes an experiment description and runs it as an ensemble.

In examples/ there are a few ready-cooked models that you can look at.

## Quick-start

* Clone this repo
* In examples/ there are a few working examples of experiments. You can download the
necessary datasets by using `bin/load_data.py`.
* Run `bin/base_model.py` for single network experiments, and `bin/ensemble.py` for ensemble
  experiments

### Datasets

Datasets are saved in the `.npz` format, with three files in a directory:
* `train.npz`: the training set
* `valid.npz`: the validation set
* `test.npz`: the test set
Each of these files is a serialised dictionary `{x: numpy.array, y: numpy.array}`
where `x` is the input data and `y` is the expected classification output.

### Experiment files

This is the file given as an argument to `base_model.py`, `ensemble.py` or
`distilled_ensemble.py`. It is a yaml description of the experiment.
Here is an example experiment file to train 10 DenseNets on CIFAR-100 using
Bagging:

```
---
## MLP Parameters ##
dataset: ../cifar-100/
data_format: npz
convert_labels_to_one_hot: true
model_file: examples/experiments/cifar-100/densenet121.model
reduce_lr_on_plateau:
  factor: 0.1
  patience: 5
  cooldown: 0
  min_lr: 0.0000001
optimizer:
  0:
    class_name: Adam
    config:
      learning_rate:
        0:  0.001
        75: 0.0001
  100:
    class_name: SGD
    config:
      learning_rate: 
        0: 0.1
        150: 0.01
        250: 0.001
      momentum: 0.9
      decay: 0.0005
epochs: 300
batch_size: 32
loss: categorical_crossentropy
shuffle: true
multi_gpu: 2

#use online image transformations by specifying arguments to ImageDataGenerator
img_gen_params:
  #zoom_range: 0.15
  width_shift_range: 0.125
  height_shift_range: 0.125
  horizontal_flip: true
  rotation_range: 15
  featurewise_std_normalization: true
  featurewise_center: true
  #zca_whitening: true

## Ensemble Parameters ##
ensemble_method: 
  class_name: Bagging
  params:
    size: 10
    aggregator: averaging
```

The parameters are as follows:

*network parameters*
 - `dataset`: the location of the dataset (format dependent).
 - `model_file`: the location of the serialised Keras model description.
 - `optimizer`: the optimization method. See separate section for
   description.
 - `epochs`: the number of training epochs.
 - `batch_size`: the number of samples to use at each iteration
 - `loss`: the loss/cost function to use. Any string accepted by Keras
   works.
 - `shuffle`: whether to shuffle the dataset at each epoch.

*ensemble parameters*
 - `ensemble_method`: the name of the Ensemble method.
 - `params`: a method-dependent set of parameters for the Ensemble.

*optimizer subparameters*
The optimizer is defined *per-epoch*. This means that in the example above, we
start with Adam and then switch to SGD at epoch 100.
 - `class_name`: a string that Keras can deserialise to a learning algorithm.
   WAME, presented at [ESANN](https://www.elen.ucl.ac.be/esann), is currently not available.
 - `config`:
   - `learning_rate`: either a float for a fixed learning rate, or a dictionary of (epoch,
     rate) pairs
   - `decay`: learning rate decay
   - `momentum`: (only valid in SGD) momentum value


*ensemble methods*
 - Bagging: Bagging
 - AdaBoostM1: AdaBoost.M1
 - DIB: [Deep Incremental Boosting](http://easychair.org/publications/paper/Deep_Incremental_Boosting).
   Parameters are as follows.
    - `n_epochs_after_first`: The number of epochs for which to train from the
      second round onwards
    - `freeze_old_layers`: `true` if the layers transferred to the next round
      are to be frozen (made not trainable)
    - `incremental_index`: the location where the new layers are to be inserted
    - `incremental_layers`: a serialized yaml of the layers to be added at each
      round

### Model files
These are standard Keras models, serialised to yaml. Effectively, this is the
verbatim output of a model's `to_yaml()`. In the examples directory you will find
both some example models, and the code that was run to generate them.
