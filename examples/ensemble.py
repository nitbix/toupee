#!/usr/bin/python
"""
Run an ensemble experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import gc
import sys
import theano
import theano.tensor as T

import mlp
import config
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
    ensemble = method.create_aggregator(params,members,x,y,train_set,valid_set)
    test_set_x, test_set_y = method.resampler.get_test()
    test_model = theano.function(inputs=[],
        on_unused_input='warn',
        outputs=ensemble.errors,
        givens={x:test_set_x, y:test_set_y})
    test_score = test_model()
    print 'Final error: {0} %'.format(test_score * 100.)
