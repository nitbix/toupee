#!/usr/bin/python

import data
import sys
import cPickle
import gzip
import os
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) > 2:
        dataset = sys.argv[1]
        fileName = sys.argv[2]
    else:
        dataset='mnist.pkl.gz'
        fileName='mnist-transformed/'
    dataset = data.load_data(dataset,shared=False)
    train,valid,test = dataset
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    aggregate_x = np.concatenate((train_x, valid_x), axis=0)
    aggregate_y = np.concatenate((train_y, valid_y), axis=0)
    t = data.Transformer((aggregate_x,aggregate_y),28,28,progress=True)
