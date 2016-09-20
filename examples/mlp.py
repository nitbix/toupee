#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import sys

from toupee import data
from toupee import config 
from toupee.mlp import sequential_model

from keras.datasets import cifar10


if __name__ == '__main__':
    params = config.load_parameters(sys.argv[1])
    dataset = data.load_data(params.dataset,
                             pickled = params.pickled,
                             one_hot_y = params.one_hot)
    mlp = sequential_model(dataset, params)
