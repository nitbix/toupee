#!/usr/bin/python

import gc
import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T
import dill

from toupee import config
from toupee.data import *

if __name__ == '__main__':
    params = config.load_parameters(sys.argv[1])
    dataset = load_data(params.dataset,
                              shared = False,
                              pickled = params.pickled)
    members = dill.load(open(sys.argv[2]))
    x = members[0].x
    y = members[0].y
    method = params.method
    method.prepare(params,dataset)
    train_set = method.resampler.get_train()
    valid_set = method.resampler.get_valid()
    ensemble = params.method.create_aggregator(params,members,x,y,train_set,valid_set)
    test_set_x, test_set_y = method.resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=ensemble.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
