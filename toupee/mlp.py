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
import keras.preprocessing.image

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

    def reshape_inputs(self,shape):
        self.train_set_x = self.orig_train_set_x.reshape([self.train_set_x.shape[0]] + shape)
        self.valid_set_x = self.orig_valid_set_x.reshape([self.valid_set_x.shape[0]] + shape)
        if self.has_test():
            self.test_set_x = self.orig_test_set_x.reshape([self.test_set_x.shape[0]] + shape)


def sequential_model(dataset, params, pretraining_set = None, model_weights = None,
        return_results = False, member_number = None, model_yaml = None,
        model_config = None, frozen_layers = [], sample_weight = None):
    """
    Initialize the parameters and create the network.
    """

    print "loading model..."
    if sample_weight is not None:
        print "using sample weights..."
    if model_config is not None:
        model = keras.models.Sequential.from_config(model_config)
    else:
        if model_yaml is None:
            with open(params.model_file, 'r') as model_file:
                model_yaml = model_file.read()
        model = keras.models.model_from_yaml(model_yaml)
    total_weights = 0

    #TODO: this count is broken for Model layers
    for w in model.get_weights():
        total_weights += numpy.prod(w.shape)

    if model_weights is not None:
        for i in range(len(model_weights)):
            model.layers[i].set_weights(model_weights[i])

    print "total weight count: {0}".format(total_weights)

    results = common.Results(params)
    data_holder = DataHolder(dataset)

    start_time = time.clock()

    metrics = ['accuracy']
    if 'additional_metrics' in params.__dict__:
        metrics = metrics + additional_metrics

    for l in frozen_layers:
        model.layers[l].trainable = False

    checkpointer = keras.callbacks.ModelCheckpointInMemory(verbose=1,
            monitor = 'val_acc',
            mode = 'max')
    callbacks = [checkpointer]

    if params.early_stopping is not None:
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=params.early_stopping['patience'], verbose=0, mode='auto')
        callbacks.append(earlyStopping)

    def callbacks_with_lr_scheduler(schedule):
        def scheduler(epoch):
            if epoch in schedule:
                print "Changing learning rate to {0}".format(schedule[epoch])
                model.optimizer.lr.set_value(schedule[epoch])
            return float(model.optimizer.lr.get_value())
        return callbacks + [keras.callbacks.LearningRateScheduler(scheduler)]

    lr_schedule = None
    if isinstance(params.optimizer['config']['lr'], dict):
        lr_schedule = params.optimizer['config']['lr']
        params.optimizer['config']['lr'] = lr_schedule[0]
    optimizer = keras.optimizers.deserialize(params.optimizer)
    model.compile(optimizer = optimizer,
                  loss = params.cost_function,
                  metrics = metrics
    )
    if params.online_transform is not None:
        def default_online_transform_param(name,default):
            if name in params.online_transform:
                return params.online_transform[name]
            else:
                return default

        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=default_online_transform_param('featurewise_center',False),
            samplewise_center=default_online_transform_param('samplewise_center',False),
            featurewise_std_normalization=default_online_transform_param('featurewise_std_normalization',False),
            samplewise_std_normalization=default_online_transform_param('samplewise_std_normalization',False),
            zca_whitening=default_online_transform_param('zca_whitening',False),
            rotation_range=default_online_transform_param('rotation_range',0),
            width_shift_range=default_online_transform_param('width_shift',0),
            height_shift_range=default_online_transform_param('height_shift',0),
            horizontal_flip=default_online_transform_param('horizontal_flip',False),
            vertical_flip=default_online_transform_param('vertical_flip',False),
            elastic_transform=default_online_transform_param('elastic_transform',None),
            pad=default_online_transform_param('pad',None),
            crop=default_online_transform_param('crop',None)
        )
        datagen.fit(data_holder.train_set_x, augment=True, rounds=2)
        pre_epochs = default_online_transform_param("after_epoch", 0)
        pre_lr = default_online_transform_param("pre_lr", params.optimizer['config']['lr'])

        #TODO: this does not work, the network is reset at every fit() call
        if pre_epochs > 0:
            print "Pre-training without transformations..."
            pre_hist = model.fit(data_holder.train_set_x, data_holder.train_set_y,
                  batch_size = params.batch_size,
                  epochs = pre_epochs,
                  validation_data = (data_holder.valid_set_x, data_holder.valid_set_y),
                  test_data = (data_holder.test_set_x, data_holder.test_set_y),
                  callbacks = callbacks_with_lr_scheduler({0: pre_lr}),
                  shuffle = params.shuffle_dataset,
                  sample_weight = sample_weight)
        print "Training with transformations..."
        if lr_schedule is not None:
            callbacks = callbacks_with_lr_scheduler(lr_schedule)
        if params.test_at_each_epoch:
            test_data = (data_holder.test_set_x, data_holder.test_set_y)
        else:
            test_data = None
        hist = model.fit_generator(
                            datagen.flow(
                                data_holder.train_set_x,
                                data_holder.train_set_y,
                                shuffle = params.shuffle_dataset,
                                batch_size = params.batch_size
                            ),
                            steps_per_epoch = data_holder.train_set_x.shape[0] / params.batch_size,
                            epochs = params.n_epochs,
                            validation_data = (data_holder.valid_set_x,
                                data_holder.valid_set_y),
                            test_data = test_data,
                            callbacks = callbacks
                           )
        if pre_epochs > 0:
            for k in pre_hist.history:
                hist.history[k] = pre_hist.history[k] + hist.history[k]
    else:
        print "Training without transformations..."
        if lr_schedule is not None:
            callbacks = callbacks_with_lr_scheduler(lr_schedule)
        hist = model.fit(data_holder.train_set_x, data_holder.train_set_y,
                  batch_size = params.batch_size,

                  epochs = params.n_epochs,
                  validation_data = (data_holder.valid_set_x, data_holder.valid_set_y),
                  test_data = (data_holder.test_set_x, data_holder.test_set_y),
                  callbacks = callbacks,
                  shuffle = params.shuffle_dataset,
                  sample_weight = sample_weight)
    model.set_weights(checkpointer.best_model)
    train_metrics = model.evaluate(data_holder.train_set_x,
            data_holder.train_set_y, batch_size = params.batch_size)
    valid_metrics = model.evaluate(data_holder.valid_set_x,
            data_holder.valid_set_y, batch_size = params.batch_size)
    if data_holder.has_test():
        test_metrics = model.evaluate(data_holder.test_set_x,
                data_holder.test_set_y, batch_size = params.batch_size)
    for metrics_name,metrics in (
            ('train', train_metrics),
            ('valid', valid_metrics),
            ('test', test_metrics)
        ):
        print "{0}:".format(metrics_name)
        for i in range(len(metrics)):
            print "  {0} = {1}".format(model.metrics_names[i], metrics[i])

    results.set_history(hist)
    end_time = time.clock()
    if data_holder.test_set_x is not None:
        print(('Optimization complete.\nBest valid accuracy: %f %%\n'
            'Obtained at epoch: %i\nTest accuracy: %f %%') %
              (valid_metrics[1] * 100.,
                  checkpointer.best_epoch + 1, test_metrics[1] * 100.))
        print('The code for ' + os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        results.set_final_observation(valid_metrics[1] * 100.,
                test_metrics[1] * 100.,
                checkpointer.best_epoch + 1)
    else:
        results.set_final_observation(valid_metrics[1]* 100., None,
                checkpointer.best_epoch + 1)
        print('Selection : Best valid score of {0} %'.format(
              valid_metrics[1] * 100.))

    if member_number is not None:
        results.member_number = member_number
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
