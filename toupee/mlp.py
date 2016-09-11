#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import gc
import os
import sys
import time
import copy
import numpy
import scipy
import math
from pymongo import MongoClient
import json

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_from_host
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.misc import imsave

import data
from data import Resampler, Transformer, sharedX
import update_rules
import layers
import config 
import cost_functions
import activations
import common
import utils

floatX = theano.config.floatX


class DataHolder:
    """
    Encapsulate the train/valid/test data to achieve a few things:
    - no leakage from multiple copies
    - ensure it is always a SharedVariable
    """

    def __init__(self,dataset):
        self.orig_train_set_x = dataset[0][0]
        self.orig_train_set_y = dataset[0][1]
        self.orig_valid_set_x = dataset[1][0]
        self.orig_valid_set_y = dataset[1][1]
        self.orig_test_set_x = dataset[2][0]
        self.orig_test_set_y = dataset[2][1]
        self.train_set_x = sharedX(self.orig_train_set_x, dtype=floatX)
        self.train_set_y = sharedX(self.orig_train_set_y, dtype='int32')
        self.valid_set_x = sharedX(self.orig_valid_set_x, dtype=floatX)
        self.valid_set_y = sharedX(self.orig_valid_set_y, dtype='int32')
        if len(dataset) > 2:
            self.test_set_x = sharedX(self.orig_test_set_x, dtype=floatX)
            self.test_set_y = sharedX(self.orig_test_set_y, dtype='int32')
        else:
            self.test_set_x, self.test_set_y = (None,None)
    
    def reset(self):
        self.train_set_x.set_value(self.orig_train_set_x)
        self.train_set_y.set_value(self.orig_train_set_y.astype('int32'))
        self.valid_set_x.set_value(self.orig_valid_set_x)
        self.valid_set_y.set_value(self.orig_valid_set_y.astype('int32'))

    def replace_shared_train(self,set_x,set_y):
        self.train_set_x = set_x
        self.train_set_y = set_y

    def set_train(self,set_x,set_y):
        self.train_set_x.set_value(set_x)
        self.train_set_y.set_value(set_y)

    def set_valid(self,set_x,set_y):
        self.valid_set_x.set_value(set_x)
        self.valid_set_y.set_value(set_y)

    def clear_train(self):
        self.train_set_x.set_value([[]])
        self.train_set_y.set_value([])

    def clear_valid(self):
        self.valid_set_x.set_value([[]])
        self.valid_set_y.set_value([])

    def clear_test(self):
        if self.test_set_x is not None:
            self.test_set_x.set_value([[]])
            self.test_set_y.set_value([])

    def clear(self):
        self.clear_train()
        self.clear_valid()
        self.clear_test()

    def has_test(self):
        return self.test_set_x is not None


class TrainingState:
    """
    Helps track the state of the current training.
    """
    
    def __init__(self,classifier):
        self.reset()
        self.classifier = classifier
        self.best_valid_loss = numpy.inf

    def reset(self):
        self.done_looping = False
        self.best_weights = None
        self.best_iter = 0
        self.best_epoch = 0
        self.test_score = None
        self.epoch = 0
        self.n_batches = {}
        self.previous_minibatch_avg_cost = 1.

    def pre_iter(self):
        self.best_weights = None
        self.best_valid_loss = numpy.inf
        self.best_iter = 0
        self.best_epoch = 0
        self.test_score = 0.
        self.epoch = 0

    def set_models(self,models):
        if 'train_error_f' in self.__dict__:
            self.train_error_f.clean_gpu()
            self.valid_error_f.clean_gpu()
            self.test_error_f.clean_gpu()
        gc.collect()
        self.train_f, self.train_error_f, self.valid_error_f, self.test_error_f = models


def sequential_model(self, params, continuation = None):
    """
    Initialize the parameters and create the network.
    """

    model = keras.models.Sequential()
    total_weights = 0
    for i,(layer_type,desc) in enumerate(self.params.n_hidden):
        if continuation is not None:
            W = continuation['W'][i]
            b = continuation['b'][i]
        else:
            W = None
            b = None

        l = make_layer(layer_type,desc,W,b,i)
        total_weights += l.weight_count
        if self.params.detailed_stats:
            print "layer {0} weight count: {1}".format(l.layer_name,l.weight_count)
        model.add(l)
        layer_number += 1

    print "total weight count: {0}".format(total_weights)
    self.rejoin_layers(input)

    if pretraining_set is not None and modes is not None:
        for mode in modes.split(','):
            self.pretrain(pretraining_set,mode)
    if continuation is not None:
        W = continuation['outW']
        b = continuation['outb']
    else:
        W = None
        b = None
    self.make_top_layer(
            self.params.n_out,
            self.chain_in,
            self.chain_n_in,
            rng,
            layer_type = self.params.output_layer,
            W = W,
            b = b,
            options = self.params.output_layer_options)


def make_layer(self,layer_type,desc,W=None,b=None,i=0):

    if(layer_type == 'flat'):
        if len(desc) == 5:
            #default no-options
            desc.append({})
        (n_this, drop_this, name_this, activation_this, weight_init, 
                options) = desc

    elif(layer_type == 'LCN'):
        n_pixels_y,n_pixels_x = self.get_n_pixels(i)
        kernel_size,use_divisor = desc

    elif(layer_type == 'LRN'):
        alpha, k, beta, n = desc

    elif(layer_type == 'elastic_transform'):
        n_pixels_y,n_pixels_x = self.get_n_pixels(i)

    elif(layer_type == 'global_pooling'):
        (mode,) = desc

    elif(layer_type == 'linear'):
        drop_this, name_this = desc

    elif(layer_type == 'pool2d'):
        if len(desc) == 3:
            #default no-options
            desc.append({})
        pooling , pool_size, name_this, options = desc

    elif layer_type in ['nin', 'mlpconv']:
        if len(desc) == 3:
            #default no-options
            desc.append({})
        n_this, drop_this, name_this, options = desc

    elif layer_type in ['convfilter', 'conv2d']:
        if len(desc) == 6:
            #default border mode
            desc.append('valid')
        if len(desc) == 7:
            #default no-options
            desc.append({})
        (input_shape,filter_shape,drop_this,name_this,
                activation_this,weight_init,border_mode,options) = desc


def make_top_layer(self, n_out, chain_in, chain_n_in, rng,
        layer_type='softmax', activation=None, name_this='temp_top',
        W = None, b = None, options = {}):
    """
    Finalize the construction by making a top layer (either to use in
    pretraining or to use in the final version)
    """
    if layer_type == 'softmax':
    if layer_type == 'logsoftmax':
    elif layer_type == 'flat':

#TODO:
# - online transform using model.fit_generator
# - load existing weights with "continuation"
# - replace MLP with a keras model
# - greedy training mode

def test_mlp(dataset, params, pretraining_set=None, x=None, y=None, index=None,
        continuation=None,return_results=False):
    results = common.Results(params)

    data_holder = DataHolder(dataset)

    rng = numpy.random.RandomState(params.random_seed)

    classifier = MLP(params=params, rng=rng, theano_rng=theano_rng, input=x,
            index=index, x=x, y=y, pretraining_set=pretraining_set,
            continuation=continuation)

    state = TrainingState(classifier)
    state.train_examples = data_holder.train_set_x.shape[0].eval()
    state.valid_examples = data_holder.valid_set_x.shape[0].eval()
    state.n_batches['train'] = state.train_examples / params.batch_size
    state.n_batches['valid'] = state.valid_examples / params.batch_size

    print "training examples: {0}".format(state.train_examples)
    print "validation examples: {0}".format(state.valid_examples)

    if data_holder.has_test():
        state.test_examples = data_holder.test_set_x.shape[0].eval()
        state.n_batches['test'] = state.test_examples / params.batch_size
        print "test examples: {0}".format(state.test_examples)

    print '... {0} training'.format(params.training_method)

    #TODO: Make these part of the YAML experiment description, after they get their own class
    # early-stopping parameters
    state.patience = 10000  # look as this many examples regardless
    state.patience_increase = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99   # a relative improvement of this much is
                                   # considered significant
    valid_frequency = min(state.n_batches['train'], state.patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the valid set; in this case we
                                  # check every epoch

    start_time = time.clock()

    model.compile(optimizer = args.update_rule,
                  loss = args.cost_function,
                  metrics = ['accuracy','cost']
    )

    def run_epoch(state,results):
        training_costs = []
        for minibatch_index in xrange(state.n_batches['train']):
            metrics = model.train_on_batch(x_batch, y_batch)
            #TODO: append to training_costs
        metrics = model.test_on_batch(x,y)
        for i in range(metrics):
            print "\t{0} = {1}\n".format(model.metrics_names[i], metrics[i])


    if params.training_method == 'normal':
        print ".... generating models"
        state.classifier.reset_hooks(state)
        state.set_models(state.classifier.make_models(data_holder))
        print ".... started"
        while (state.epoch < params.n_epochs) and (not state.done_looping):
            epoch_start = time.clock()
            state.epoch += 1
            run_epoch(state,results)
            epoch_end = time.clock()
            print "t: {0}".format(epoch_end - epoch_start)
        if state.best_weights is not None:
            state.classifier.set_weights(state.best_weights)

    end_time = time.clock()
    if data_holder.test_set_x is not None:
        print(('Optimization complete. Best valid score of %f %% '
               'obtained at iteration %i, epoch %i, with test performance %f %%') %
              (state.best_valid_loss * 100., state.best_iter + 1,
                  state.best_epoch, state.test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        results.set_final_observation(state.best_valid_loss * 100., state.test_score * 100., state.best_epoch)
    else:
        results.set_final_observation(state.best_valid_loss * 100., None, state.best_epoch)
        print('Selection : Best valid score of {0} %'.format(
              state.best_valid_loss * 100.))
    cl = state.classifier
    if params.online_transform is None:
        #if we have online transforms, this has already been deleted
        state.train_error_f.clean_gpu()
    state.valid_error_f.clean_gpu()
    if 'results_db' in params.__dict__ :
        if 'results_host' in params.__dict__:
            host = params.results_host
        else:
            host = None
        conn = MongoClient(host=host)
        db = conn[params.results_db]
        if 'results_table' in params.__dict__: 
            table_name = params.results_table
        else:
            table_name = 'results'
        table = db[table_name]
        print "saving results to {0}@{1}:{2}".format(params.results_db,host,table)
        table.insert(json.loads(json.dumps(results.__dict__,default=common.serialize)))
    if return_results:
        return cl,results
    else:
        return cl
