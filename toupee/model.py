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
#TODO: sample weights
#TODO: early stopping
#TODO: checkpointing?


class OptimizerSchedulerCallback(tf.keras.callbacks.Callback):

    def __init__(self, optimizer_schedule):
        super(OptimizerSchedulerCallback, self).__init__()
        self.optimizer_schedule = optimizer_schedule

    def on_epoch_end(self, epoch, logs=None):
        """ Callback to stop training and change the optimizer """
        epoch_keys = self.optimizer_schedule.params.keys()
        if epoch + 1 in epoch_keys:
            optimizer = self.optimizer_schedule[epoch+1]
            self.model.compile(optimizer, loss=self.model.loss, metrics=self.model.metrics)


class OptimizerSchedule:
    """ Schedules Optimizers and Learning Rates according to config """
    def __init__(self, params, epochs):
        """ Create an optimizer from params and learning rate """
        self.params = copy.deepcopy(params)
        self.optimizers = {}
        self.epochs = epochs
        if 'class_name' in self.params: # Force this to be an epoch schedule even if it's not
            self.params = {0: self.params}
        for thresh, opt_params in self.params.items():
            conf = copy.deepcopy(opt_params)
            if isinstance(conf['config']['learning_rate'], dict):
                lr = conf['config']['learning_rate']
                conf['config']['learning_rate'] = lr[min(lr.keys())]
            self.optimizers[thresh] = tf.keras.optimizers.deserialize(conf)
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(self._lr_scheduler)
    
    def _params_scheduler(self, epoch):
        for thresh in sorted(self.params.keys(), reverse=True):
            if epoch >= thresh:
                return self.params[thresh]

    def _opt_scheduler(self, epoch):
        for thresh in sorted(self.optimizers.keys(), reverse=True):
            if epoch >= thresh:
                return self.optimizers[thresh]
    
    def __getitem__(self, epoch):
        """
        Subscripting operator, so we can use [] to get the right
        optimizer for any epoch
        """ 
        return self._opt_scheduler(epoch)

    def _lr_scheduler(self, epoch):
        #TODO: use sorting
        lr = None
        params = self._params_scheduler(epoch)
        if isinstance(params['config']['learning_rate'], dict):
            for thresh, value in params['config']['learning_rate'].items():
                if epoch >= thresh:
                    lr = value
        else:
            lr = params['config']['learning_rate']
        return lr

    def get_callbacks(self, loss, metrics):
        return [self.lr_callback,
                OptimizerSchedulerCallback(self)]


class Model:
    """ Representation of a model """
    #TODO: Frozen layers
    #TODO: Get model id and use different tb log dir for each model
    def __init__(self, params):
        self.params = params
        with open(params.model_file, 'r') as model_file:
            model_yaml = model_file.read()
        self._model = tf.keras.models.model_from_yaml(model_yaml)
        if params.model_weights:
            self._model.load_weights(params.model_weights)
        self._optimizer_schedule = OptimizerSchedule(params.optimizer, self.params.epochs)
        self._loss = tf.keras.losses.deserialize(params.loss)
        self.params = params
        self._training_metrics = ['accuracy']

    def fit(self, data, verbose=None):
        """ Train a model """
        start_time = time.clock()
        callbacks = self._optimizer_schedule.get_callbacks(self._loss,
                                                           self._training_metrics)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.params.tb_log_dir))
        if self.params.reduce_lr_on_plateau:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(**self.params.reduce_lr_on_plateau))
        if self.params.multi_gpu:
            print("!!! WARNING - EXPERIMENTAL !!! running on multi gpu")
            tf.keras.utils.multi_gpu_model(self._model, gpus=self.params.multi_gpu)
        self.img_gen = data.img_gen
        self._model.compile(
            optimizer = self._optimizer_schedule[0],
            loss = self._loss,
            metrics = self._training_metrics,
            )
        self._model.fit(
            data.get_training_handle(),
            epochs = self.params.epochs,
            steps_per_epoch = data.steps_per_epoch['train'],
            shuffle = 'batch',
            callbacks = callbacks,
            verbose = verbose or self.params.verbose,
            validation_data = data.get_validation_handle(standardized=True),
            )
        end_time = time.clock()
        print('Model trained for %.2fm' % ((end_time - start_time) / 60.))
        self.test_metrics = self.evaluate(data.get_testing_handle())

    def evaluate(self, test_data):
        """ Evaluate model on some test data handle """
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
        if self.img_gen:
            X = self.img_gen.standardize(X)
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
