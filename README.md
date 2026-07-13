# Welcome to Toupee
![Toupee Logo](/logo.png)

"The ugly thing on top that covers up what's missing"

A library for Deep Learning ensembles, with a toolkit for running reproducible
experiments, built on PyTorch.

Toupee was developed to support research on Deep Learning Ensembles:
incremental construction, knowledge distillation, and calibration. It began
life in 2015 on Theano, later moved to a Keras/TensorFlow backend, and was
redesigned on PyTorch in 2026.

## What it does

* Describes a complete experiment in a single declarative YAML file: dataset,
  model, optimiser schedule, and ensemble method
* Trains ensembles with several methods:
  * `Bagging`
  * `AdaBoost`
  * `Incremental` (Simple Incremental Ensembles)
  * `DIB` ([Deep Incremental Boosting](http://easychair.org/publications/paper/Deep_Incremental_Boosting))
  * `DIBag` (Deep Incremental Bagging)
* Distils a trained ensemble into a single network (`--distil`)
* Evaluates calibration (classwise Expected Calibration Error) and
  adversarial robustness (`--adversarial-testing`)
* Logs to TensorBoard (`--tensorboard`) or Weights & Biases (`--wandb`)

## Quick-start

* Clone this repo and install the requirements:
  `pip install -r requirements.txt`
* Download the example datasets with `bin/load_data.py`. Datasets are stored
  as `.npz` files in a directory (`train.npz`, `valid.npz`, `test.npz`), each
  a serialised dictionary `{x: numpy.array, y: numpy.array}`
* Run a single-network experiment:
  `python bin/base_model.py examples/experiments/mnist/vanilla.yaml`
* Run an ensemble experiment:
  `python bin/ensemble.py examples/experiments/mnist/vanilla.yaml`

In `examples/experiments/` there are ready-made configurations for MNIST,
CIFAR-10 and CIFAR-100, including the ensemble methods listed above.

### Experiment files

An experiment is a YAML description of the model, the training schedule, and
the ensemble method. For example:

```yaml
## MLP Parameters ##
dataset: /local/mnist_th/
model_file: cnn.model
optimizer:
  class_name: WAME
  config:
    lr:
      0:  0.001
      10: 0.0001
      20: 0.00001
n_epochs: 50
batch_size: 128
cost_function: categorical_crossentropy
shuffle_dataset: true

## Ensemble Parameters ##
resample_size: 60000
method: !Bagging {}
ensemble_size: 10
```

The optimiser is defined per-epoch, so schedules can switch method or
learning rate at any epoch. WAME (presented at
[ESANN 2017](https://www.elen.ucl.ac.be/esann)) is available alongside the
standard PyTorch optimisers.

### Command-line options

`bin/ensemble.py` accepts:

 - `--epochs N` and `--size N`: override the number of epochs / ensemble size
 - `--distil`: distil the trained ensemble into a single network
 - `--adversarial-testing`: evaluate robustness to adversarial examples
 - `--tensorboard`: log metrics to TensorBoard
 - `--wandb` (with `--wandb-project` and `--wandb-group`): log to
   Weights & Biases

## History and citing

Toupee was created by Alan Mosca at Birkbeck, University of London, to support
the experiments in his PhD thesis, *Incremental Construction of Deep Learning
Ensembles: Efficient Training, Distillation and Calibration*. If you use
Toupee in your research, please cite:

```bibtex
@article{mosca2018customised,
  title={Customised ensemble methodologies for deep learning:
         Boosted Residual Networks and related approaches},
  author={Mosca, Alan and Magoulas, George D},
  journal={Neural Computing and Applications},
  year={2019},
  publisher={Springer}
}
```

## License

MIT — see [LICENSE](LICENSE).
