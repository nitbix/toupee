#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time
import copy
import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams                                                                                                                    
from toupee.logistic_sgd import LogisticRegression
from toupee import data
from toupee.data import Resampler, Transformer, sharedX
from toupee import update_rules
from toupee import layers
from toupee import config 
from toupee import cost_functions
from toupee.mlp import MLP, test_mlp


if __name__ == '__main__':
    #turn this on only if you want to do parameter search
    search_epochs = 40
    search = False

    params = config.load_parameters(sys.argv[1])
    dataset = data.load_data(params.dataset,
                              shared = True,
                              pickled = params.pickled)
    pretraining_set = data.make_pretraining_set(dataset,params.pretraining)
    if not search:
        mlp = test_mlp(dataset, params, pretraining_set = pretraining_set)
    else:
        params.n_epochs = search_epochs
        for eta_minus in [0.01,0.1,0.5,0.75,0.9]:
            params.update_rule.eta_minus = eta_minus
            for eta_plus in [1.001,1.01,1.1,1.2,1.5]:
                params.update_rule.eta_plus = eta_plus
                for min_delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                    params.update_rule.min_delta = min_delta
                    for max_delta in [50]:
                        print "PARAMS:"
                        print "ETA-: {0}".format(eta_minus)
                        print "ETA+: {0}".format(eta_plus)
                        print "MIN_DELTA: {0}".format(min_delta)
                        print "MAX_DELTA: {0}".format(max_delta)
                        params.update_rule.max_delta = max_delta
                        try:
                            mlp=test_mlp(dataset, params, pretraining_set = pretraining_set)
                        except KeyboardInterrupt:
                            print "skipping manually to next"
