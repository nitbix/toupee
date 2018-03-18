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
    x = T.matrix('x')
    y = T.ivector('y')
    method = params.method
    method.prepare(params,dataset)
    train_set = method.resampler.get_train()
    valid_set = method.resampler.get_valid()
    members = []
    for i in range(0,params.ensemble_size):
        print('training member {0}'.format(i))
        new_member = method.create_member(x,y)
        members.append(new_member)
        gc.collect()
    dill.dump(members,open(sys.argv[2],"wb"))
