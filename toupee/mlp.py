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

import data
from data import Resampler, Transformer
import config 
import common
import utils

import keras

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
        self.train_set_x = self.orig_train_set_x
        self.train_set_y = self.orig_train_set_y
        self.valid_set_x = self.orig_valid_set_x
        self.valid_set_y = self.orig_valid_set_y
        if len(dataset) > 2:
            self.test_set_x = self.orig_test_set_x
            self.test_set_y = self.orig_test_set_y
        else:
            self.test_set_x, self.test_set_y = (None,None)
    
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


def sequential_model(dataset, params, pretraining_set = None, model_weights = None,
        return_results = False):
    """
    Initialize the parameters and create the network.
    """
#TODO:
# - online transform using model.fit_generator
# - greedy training mode


    print "loading model..."
    with open(params.model_file, 'r') as model_file:
        model_yaml = model_file.read()
    model = keras.models.model_from_yaml(model_yaml)
    total_weights = 0
    if model_weights is not None:
        model.set_weights(model_weights)

    #TODO: weight count
    print "total weight count: {0}".format(total_weights)

    results = common.Results(params)
    data_holder = DataHolder(dataset)

    rng = numpy.random.RandomState(params.random_seed)

    state = TrainingState(model)
    state.train_examples = data_holder.train_set_x.shape[0]
    state.valid_examples = data_holder.valid_set_x.shape[0]
    state.n_batches['train'] = state.train_examples / params.batch_size
    state.n_batches['valid'] = state.valid_examples / params.batch_size

    print "training examples: {0}".format(state.train_examples)
    print "validation examples: {0}".format(state.valid_examples)

    if data_holder.has_test():
        state.test_examples = data_holder.test_set_x.shape[0]
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

    metrics = ['accuracy']
    if 'additional_metrics' in params.__dict__:
        metrics = metrics + additional_metrics

    model.compile(optimizer = params.update_rule,
                  loss = params.cost_function,
                  metrics = metrics
    )

    def run_epoch(state,results):
        print "epoch {0}".format(state.epoch)
        training_costs = []
        if params.all_in_memory:
            model.train_on_batch(data_holder.train_set_x,
                    data_holder.train_set_y,
                    batch_size = params.batch_size)
        else:
            for minibatch_index in xrange(state.n_batches['train']):
                batch_start = minibatch_index * params.batch_size
                batch_end = max((minibatch_index + 1) * params.batch_size, state.train_examples )
                batch_metrics = model.train_on_batch(
                            data_holder.train_set_x[batch_start:batch_end],
                            data_holder.train_set_y[batch_start:batch_end]
                        )
                sys.stdout.write("\r  batch {0}, cost: {1}".format(minibatch_index,
                    batch_metrics[0]))
                sys.stdout.flush()
                #TODO: append to training_costs
            sys.stdout.write("\n")
        train_metrics = model.test_on_batch(data_holder.train_set_x,data_holder.train_set_y)
        valid_metrics = model.test_on_batch(data_holder.valid_set_x,data_holder.valid_set_y)
        if data_holder.has_test():
            test_metrics = model.test_on_batch(data_holder.test_set_x,data_holder.test_set_y)
        for metrics_name,metrics in (
                ('train', train_metrics),
                ('valid', valid_metrics),
                ('test', test_metrics)
            ):
            print "{0}:".format(metrics_name)
            for i in range(len(metrics)):
                print "  {0} = {1}".format(model.metrics_names[i], metrics[i])
#TODO HERE:
# - add the best values to state
# - save best weights
# - check early stopping 


    if params.training_method == 'normal':
        print "started training"
        while (state.epoch < params.n_epochs) and (not state.done_looping):
            epoch_start = time.clock()
            state.epoch += 1
            run_epoch(state,results)
            epoch_end = time.clock()
            print "t: {0}".format(epoch_end - epoch_start)
        if state.best_weights is not None:
            state.model.set_weights(state.best_weights)

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
        return model, results
    else:
        return model
