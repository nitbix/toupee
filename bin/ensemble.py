#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T

import mlp
import config
from logistic_sgd import LogisticRegression
from data import *

if __name__ == '__main__':
    params = config.load_parameters(sys.argv[1])
    dataset = load_data(params.dataset,
                              shared = False,
                              pickled = params.pickled)
    x = T.matrix('x')
    y = T.ivector('y')
    method = params.method
    method.prepare(dataset)
    train_set = method.resampler.get_train()
    valid_set = method.resampler.get_valid()
    members = []
    for i in range(0,params.ensemble_size):
        print 'training member {0}'.format(i)
        new_member = method.create_member(x,y)
        members.append(new_member)
        gc.collect()
    ensemble = params.method.create_aggregator(params,members,x,y,train_set,valid_set)
    test_set_x, test_set_y = resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=ensemble.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
