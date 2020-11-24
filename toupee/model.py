#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import time
import copy
import logging
import tensorflow as tf # type: ignore
import numpy as np # type: ignore

import toupee as tp
#TODO: backprop to the inputs
#TODO: sample weights
#TODO: early stopping
#TODO: checkpointing?


class TrainLossStore(tf.keras.callbacks.Callback):
    def __init__(self, folder, data_unshuffled, model_cls_instance):
        super().__init__()
        self.store_folder = os.path.join(folder, "train_loss")
        os.makedirs(self.store_folder, exist_ok=True)
        self.data_unshuffled = data_unshuffled
        self.model_cls_instance = model_cls_instance

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch} end. ", end="", flush=True)
        all_y_pred_proba = []
        all_y_true = []
        print("Getting train predictions. ", end="", flush=True)
        asserted = False
        for (x, y_true) in self.data_unshuffled.get_handle("train", standardized=True):
            if not asserted:
                # To assert that the data is NOT shuffled, each epoch. ##paranoia
                # If data parameters like batch size change, this might change.
                assert np.argmax(y_true[2]) == 0
                assert np.isclose(x[31, 31, 31, 0], 0.42758296500872006)
                asserted = True
            all_y_pred_proba.append(self.model_cls_instance.predict_proba(x))
            all_y_true.append(np.argmax(y_true.numpy(), axis=1))
        y_pred_proba = np.concatenate(all_y_pred_proba)
        y_true = np.concatenate(all_y_true)
        print(f"Getting loss on {y_true.shape[0]} samples. ", end="", flush=True)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        all_losses = scce(y_true, y_pred_proba).numpy()
        file_name = os.path.join(self.store_folder, f"epoch_{epoch}.npz")
        current_lr = self.model_cls_instance._model.optimizer.lr.numpy()
        print(f"Storing in {file_name}.")
        np.savez(file_name, loss=all_losses, current_lr=current_lr)


class OptimizerSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, optimizer_schedule, metrics):
        super(OptimizerSchedulerCallback, self).__init__()
        self.optimizer_schedule = optimizer_schedule
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs=None):
        """ Callback to stop training and change the optimizer """
        epoch_keys = self.optimizer_schedule.params.keys()
        if epoch + 1 in epoch_keys:
            optimizer = self.optimizer_schedule[epoch+1]
            self.model.compile(optimizer, loss=self.model.loss, metrics=self.metrics)


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
                OptimizerSchedulerCallback(self, metrics=metrics)]


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
        #self._loss = tf.keras.losses.deserialize(params.loss)
        self._loss = params.loss
        self.params = params
        self._training_metrics = ['accuracy']

    def fit(self, data, verbose=None):
        """ Train a model """
        start_time = time.perf_counter()
        callbacks = self._optimizer_schedule.get_callbacks(self._loss,
                                                           self._training_metrics)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.params.tb_log_dir))
        if self.params.reduce_lr_on_plateau:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(**self.params.reduce_lr_on_plateau))
        if self.params.multi_gpu:
            logging.warning("\n\n!!! WARNING - EXPERIMENTAL !!! running on multi gpu\n\n")
            tf.keras.utils.multi_gpu_model(self._model, gpus=self.params.multi_gpu)
        if self.params.override_train_step:
            self._override_train_step()
        self._model.compile(
            optimizer=self._optimizer_schedule[0],
            loss=self._loss,
            metrics=self._training_metrics,
            )
        if self.params.store_train_loss:
            logging.warning("\n\n!!! WARNING - EXPERIMENTAL !!! storing train loss for each sample at epoch end\n\n")
            param_copy = self.params.__dict__.copy()
            param_copy["shuffle"] = False
            data_unshuffled = tp.data.Dataset(src_dir=param_copy["dataset"], **param_copy)
            callbacks.append(TrainLossStore(
                folder="/home/joao/research/densenet_test_loss",
                data_unshuffled=data_unshuffled,
                model_cls_instance=self
            ))
        self._model.fit(
            data.get_training_handle(),
            epochs=self.params.epochs,
            steps_per_epoch=data.steps_per_epoch['train'],
            shuffle='batch',
            callbacks=callbacks,
            verbose=verbose or self.params.verbose,
            validation_data=data.get_validation_handle(standardized=True),
            )
        end_time = time.perf_counter()
        logging.info('Model trained for %.2fm' % ((end_time - start_time) / 60.))
        self.test_metrics = self.evaluate(data.get_testing_handle(standardized=True))

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
        return self._model.predict(X)

    def predict_classes(self, X):
        """ Aggregated argmax """
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, filename):
        """ Train a model """
        self._model.save(filename)

    def get_keras_model(self):
        """ Return raw Keras model """
        return self._model

    def _override_train_step(self):
        """ Overrides the train step so as to expose a few internal variables"""
        logging.warning("\n\n!!! WARNING - EXPERIMENTAL !!! overriding model train step\n\n")

        from tensorflow.python.eager import backprop
        from tensorflow.python.keras.engine import data_adapter
        from tensorflow.python.keras.engine.training import _minimize

        def train_step(self, data):
            """The logic for one training step.
            This method can be overridden to support custom training logic.
            This method is called by `Model.make_train_function`.
            This method should contain the mathemetical logic for one step of training.
            This typically includes the forward pass, loss calculation, backpropagation,
            and metric updates.
            Configuration details for *how* this logic is run (e.g. `tf.function` and
            `tf.distribute.Strategy` settings), should be left to
            `Model.make_train_function`, which can also be overridden.
            Arguments:
            data: A nested structure of `Tensor`s.
            Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
            """
            # These are the only transformations `Model.fit` applies to user-input
            # data when a `tf.data.Dataset` is provided. These utilities will be exposed
            # publicly.
            data = data_adapter.expand_1d(data)
            x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

            with backprop.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(
                    y, y_pred, sample_weight, regularization_losses=self.losses)
            # For custom training steps, users can just write:
            #   trainable_variables = self.trainable_variables
            #   gradients = tape.gradient(loss, trainable_variables)
            #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            # The _minimize call does a few extra steps unnecessary in most cases,
            # such as loss scaling and gradient clipping.
            _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                    self.trainable_variables)

            self.compiled_metrics.update_state(y, y_pred, sample_weight)
            return {m.name: m.result() for m in self.metrics}

        # redefines the method of the instance `_model`
        import types
        self._model.train_step = types.MethodType(train_step, self._model)
