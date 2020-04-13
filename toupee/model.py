#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import time
import copy
import tensorflow as tf
import numpy as np

import toupee as tp
#TODO: backprop to the inputs

class Model:
    """ Representation of a model """
    #TODO: Frozen layers
    def __init__(self, params):
        self.params = params
        with open(params.model_file, 'r') as model_file:
            model_yaml = model_file.read()
        self._model = tf.keras.models.model_from_yaml(model_yaml)
        if params.model_weights:
            self._model.load_weights(params.model_weights)
        self.learning_rate = params.optimizer['config']['learning_rate']
        optimizer_params = copy.deepcopy(params.optimizer)
        if isinstance(self.learning_rate, dict):
            optimizer_params['config']['learning_rate'] = self.learning_rate[0]
        self._optimizer = tf.keras.optimizers.deserialize(optimizer_params)
        self.params = params
        self._metrics = ['accuracy']
        self._model.compile(optimizer = self._optimizer,
                      loss = params.cost_function,
                      metrics = self._metrics,
                     )

    def fit(self, data, verbose=None):
        """ Train a model """
        #TODO: validation set
        #TODO: test metrics
        start_time = time.clock()
        def _lr_scheduler(epoch):
            lr = None
            if isinstance(self.learning_rate, dict):
                for thresh, value in self.learning_rate.items():
                    if epoch >= thresh:
                        lr = value
            else:
                lr = self.learning_rate
            return lr
        lr_callback = tf.keras.callbacks.LearningRateScheduler(_lr_scheduler)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.params.tb_log_dir), lr_callback] #TODO: TensorBoard

        self._model.fit(
                data.get_training_handle(),
                epochs = self.params.epochs,
                shuffle = 'batch',
                callbacks = callbacks,
                verbose = verbose or self.params.verbose,
                #TODO: this is currently broken because tf can't handle it...
                #validation_data = data.get_validation_handle(),
                )
        end_time = time.clock()
        print('Model trained for %.2fm' % ((end_time - start_time) / 60.))
        self.test_metrics = self.evaluate(data.get_testing_handle())

    def evaluate(self, test_data):
        """ Evaluate model on some test data """
        #TODO: update for different data formats
        all_y_pred = []
        all_y_true = []
        for (x, y_true) in test_data:
            all_y_pred.append(self.predict_classes(x))
            all_y_true.append(np.argmax(y_true.numpy(), axis=1))
        y_pred = np.concatenate(all_y_pred)
        y_true = np.concatenate(all_y_true)
        return tp.utils.eval_scores(y_true, y_pred)

    def predict_proba(self, X):
        """ Output logits """
        return self._model.predict(X)

    def predict_classes(self, X):
        """ Aggregated argmax """
        return np.argmax(self.predict_proba(X), axis = 1)

    def save(self, filename):
        """ Train a model """
        self._model.save(filename)

    def get_keras_model(self):
        """ Return raw Keras model """
        return self._model

# import gc
# import os
# import sys
# import time
# import copy
# import numpy
# import scipy
# import math
# import json
# import random

# from pymongo import MongoClient

# from toupee.data import Resampler, Transformer
# import toupee.config as config
# import toupee.common as common
# import toupee.utils as utils

# import keras
# import keras.preprocessing.image
# from keras import backend as K


# def initialize_model(params, sample_weight, model_config, model_yaml,
#                         model_weights, frozen_layers):
#     print("loading model...")
#     if sample_weight is not None:
#         print("using sample weights...")
#     if model_config is not None:
#         model = keras.models.Sequential.from_config(model_config)
#     else:
#         if model_yaml is None:
#             with open(params.model_file, 'r') as model_file:
#                 model_yaml = model_file.read()
#         model = keras.models.model_from_yaml(model_yaml)
#     total_weights = 0

#     #TODO: this count is broken for Model layers
#     for w in model.get_weights():
#         total_weights += numpy.prod(w.shape)

#     if model_weights is not None:
#         for i in range(len(model_weights)):
#             model.layers[i].set_weights(model_weights[i])

#     print(("total weight count: {0}".format(total_weights)))
#     if frozen_layers is None:
#         frozen_layers = []
#     for l in frozen_layers:
#         model.layers[l].trainable = False
#     return(model, total_weights)


# def initialize_metrics(params):
#     if params.classification == True:
#         scorer_name = 'accuracy'
#         monitor_type = 'val_acc'
#     else:
#         scorer_name = 'mean_squared_error'
#         monitor_type = 'val_loss'
#     metrics = [scorer_name]
#     if 'additional_metrics' in params.__dict__:
#         metrics = metrics + additional_metrics

#     checkpointer = common.ModelCheckpointInMemory(verbose=1,
#             monitor = monitor_type,
#             mode = 'max')
#     return(metrics, checkpointer)

# def callbacks_with_lr_scheduler(schedule, model, callbacks):
#     def scheduler(epoch):
#         if epoch in schedule:
#             print(("Changing learning rate to {0}".format(schedule[epoch])))
#             # model.optimizer.lr.set_value(schedule[epoch])     #<--- old keras-fork version
#             model.optimizer.lr = K.variable(value = schedule[epoch])
#         # return float(model.optimizer.lr.get_value())          #<--- old keras-fork version
#         return float(K.eval(model.optimizer.lr))
#     return callbacks + [keras.callbacks.LearningRateScheduler(scheduler)]

# def print_results(model, train_metrics, valid_metrics, test_metrics):
#     for metrics_name,metrics in (
#             ('train', train_metrics),
#             ('valid', valid_metrics),
#             ('test', test_metrics)
#         ):
#         print(("{0}:".format(metrics_name)))
#         for i in range(len(metrics)):
#             print(("  {0} = {1}".format(model.metrics_names[i], metrics[i])))

# def sequential_model(dataset,
#                      params,
#                      pretraining_set = None,
#                      model_weights = None,
#                      return_results = False,
#                      member_number = None,
#                      model_yaml = None,
#                      model_config = None,
#                      frozen_layers = None,
#                      sample_weight = None,
#                      ):
#     """
#     Initialize the parameters and create the network.
#     [GENERATOR DATA VERSION]
#     """

#     #_ was "total_weights" before
#     model, _ = initialize_model(params, sample_weight, model_config,
#                                             model_yaml, model_weights, frozen_layers)

#     if return_results:
#         results = common.Results(params)
#     #3-4 data holders: (1) sampled train data, (2-3) eval data - train/valid/[test] sets
#     sampled_indexes = dataset[0][0]
#     if sampled_indexes is not None:
#         sampled_indexes.sort()
#     files = dataset[1]
#     train_holder = files[0]
#     test_holder = files[2]
#     start_time = time.clock()
#     metrics, checkpointer = initialize_metrics(params)
#     callbacks = [checkpointer]

#     if params.early_stopping is not None:
#         earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',
#             patience=params.early_stopping['patience'], verbose=0, mode='auto')
#         callbacks.append(earlyStopping)

#     lr_schedule = None
#     if isinstance(params.optimizer['config']['lr'], dict):
#         lr_schedule = params.optimizer['config']['lr']
#         params.optimizer['config']['lr'] = lr_schedule[0]
#     optimizer = keras.optimizers.deserialize(params.optimizer)
#     model.compile(optimizer = optimizer,
#                   loss = params.cost_function,
#                   metrics = metrics,
#                  )

#     #TODO - Joao: I think this if branch needs to be updated with the new data holder
#     if params.online_transform is not None:
#         raise NotImplementedException()
#         #check the bottom of this file for the old code
#     else:
#         print("Training without transformations...")
#         print('Verbosity level:', params.verbose)
#         if lr_schedule is not None:
#             callbacks = callbacks_with_lr_scheduler(lr_schedule, model, callbacks)
#         if return_results:
#             hist = model.fit(
#                             x = train_holder['X'],
#                             y = train_holder['y'],
#                             epochs = params.n_epochs,
#                             batch_size = params.batch_size,
#                             shuffle = 'batch',
#                             validation_split=0.1,
#                             callbacks = callbacks,
#                             verbose = params.verbose,
#                             )
#         else:
#             model.fit(
#                     x = train_holder['X'],
#                     y = train_holder['y'],
#                     epochs = params.n_epochs,
#                     batch_size = params.batch_size,
#                     shuffle = 'batch',
#                     validation_split=0.1,
#                     callbacks = callbacks,
#                     verbose = params.verbose,
#                     )

#     model.set_weights(checkpointer.best_model)
#     #evals everything with a generator
#     print('\nGetting the train metrics...')
#     train_metrics = model.evaluate(
#                                 x=train_holder['X'],
#                                 y=train_holder['y'],
#                                 batch_size=params.batch_size,
#                                   )
#     print('Getting the validation metrics...')
#     valid_metrics = train_metrics # This is just here until we decide whether to keep a validation
#                                   # dataset 
#     print('Getting the test metrics...')
#     test_metrics = model.evaluate(
#                                 x=test_holder['X'],
#                                 y=test_holder['y'],
#                                 batch_size=params.batch_size,
#                                  )
#     print_results(model, train_metrics, valid_metrics, test_metrics)

#     if return_results:
#         results.set_history(hist)
#     end_time = time.clock()
#     print((('Optimization complete.\nBest valid: %f \n'
#         'Obtained at epoch: %i\nTest: %f ') %
#           (valid_metrics[1],
#               checkpointer.best_epoch + 1, test_metrics[1])))
#     print(('The code for ' + os.path.split(__file__)[1] +
#                           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
#     if return_results:
#         results.set_final_observation(valid_metrics[1],
#             test_metrics[1],
#             checkpointer.best_epoch + 1)

#     if (member_number is not None) and (return_results):
#         results.member_number = member_number

#     if return_results:
#         return model, results
#     else:
#         return model

# #------------------------------------------------------------------------------------------        
# # code to update later:  
#         # def default_online_transform_param(name,default):
#             # if name in params.online_transform:
#                 # return params.online_transform[name]
#             # else:
#                 # return default

#         # datagen = keras.preprocessing.image.ImageDataGenerator(
#             # featurewise_center=default_online_transform_param('featurewise_center',False),
#             # samplewise_center=default_online_transform_param('samplewise_center',False),
#             # featurewise_std_normalization=default_online_transform_param('featurewise_std_normalization',False),
#             # samplewise_std_normalization=default_online_transform_param('samplewise_std_normalization',False),
#             # zca_whitening=default_online_transform_param('zca_whitening',False),
#             # rotation_range=default_online_transform_param('rotation_range',0),
#             # width_shift_range=default_online_transform_param('width_shift',0),
#             # height_shift_range=default_online_transform_param('height_shift',0),
#             # horizontal_flip=default_online_transform_param('horizontal_flip',False),
#             # vertical_flip=default_online_transform_param('vertical_flip',False),
#             # elastic_transform=default_online_transform_param('elastic_transform',None),
#             # pad=default_online_transform_param('pad',None),
#             # crop=default_online_transform_param('crop',None)
#         # )
#         # datagen.fit(data_holder.train_set_x, rounds=1)
#         # pre_epochs = default_online_transform_param("after_epoch", 0)
#         # pre_lr = default_online_transform_param("pre_lr", params.optimizer['config']['lr'])

#         # TODO: this does not work, the network is reset at every fit() call
#         # if pre_epochs > 0:
#             # print("Pre-training without transformations...")
#             # pre_hist = model.fit(data_holder.train_set_x, data_holder.train_set_y,
#                   # batch_size = params.batch_size,
#                   # epochs = pre_epochs,
#                   # validation_data = (data_holder.valid_set_x, data_holder.valid_set_y),
#                   # test_data = (data_holder.test_set_x, data_holder.test_set_y),
#                   # callbacks = callbacks_with_lr_scheduler({0: pre_lr}, model, callbacks),
#                   # shuffle = params.shuffle_dataset,
#                   # sample_weight = sample_weight)
#         # print("Training with transformations...")
#         # if lr_schedule is not None:
#             # callbacks = callbacks_with_lr_scheduler(lr_schedule, model, callbacks)
#         # if params.test_at_each_epoch:
#             # test_data = (data_holder.test_set_x, data_holder.test_set_y)
#         # else:
#             # test_data = None
#         # hist = model.fit_generator(
#                             # datagen.flow(
#                                 # data_holder.train_set_x,
#                                 # data_holder.train_set_y,
#                                 # shuffle = params.shuffle_dataset,
#                                 # batch_size = params.batch_size
#                             # ),
#                             # steps_per_epoch = data_holder.train_set_x.shape[0] / params.batch_size,
#                             # epochs = params.n_epochs,
#                             # validation_data = (data_holder.valid_set_x,
#                                 # data_holder.valid_set_y),
#                             # test_data = test_data,
#                             # callbacks = callbacks
#                            # )
#         # if pre_epochs > 0:
#             # for k in pre_hist.history:
#                 # hist.history[k] = pre_hist.history[k] + hist.history[k]
