#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import time
import logging
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import pandas as pd # type: ignore
import math
import uuid
import wandb

import toupee as tp

#TODO: AdaBoost MA
#TODO: BRN
#TODO: BaRN
#TODO: Snapshot
#TODO: ManyModels


class EnsembleMethod:
    """ Abstract representation of an Ensemble from which all other methods are derived """
    def __init__(self, data, size, model_params, aggregator, model_factory=tp.model.Model, saved_ensemble=None, wandb=None,
                 adversarial_testing=False, distil=False, **kwargs):
        self.data = data
        self.size = size
        self.model_params = model_params
        self.members = []
        self.aggregator = tp.ensembles.get_aggregator(aggregator)
        self.model_factory = model_factory
        self.wandb = wandb
        self.distil = distil
        self.adversarial_testing = adversarial_testing
        self.model_weights = [1. / float(self.size) for _ in range(self.size)]
        self._fit_loop_info = {
            'current_step': None,
            'current_model': None,
        }
        if kwargs:
            logging.warning("Unknown ensemble parameters: %s" % kwargs)
        # contract: derived classes must set the members list or generator
        if saved_ensemble:
            self._load(saved_ensemble)
        else:
            self._initialise_members()

    def _initialise_members(self):
        """ Abstract - must be implemented to initialise the members array """
        raise NotImplementedError()

    def _default_value(self, param_name, value):
        if param_name not in self.__dict__:
            logging.warning(("setting default for: {0} to {1}" \
            .format(param_name, value)))
            self.__dict__[param_name] = value

    def _members(self):
        """
        Generator function that yield the members one at a time. This version
        is based off of a static members list, but can be modified to create new
        members parametrically.
        """
        for member in self.members:
            yield member

    def save(self, location):
        """ Saves an ensemble """
        # for m in members:
        #     m.save(...)
        # _save_ensemble_details
        raise NotImplementedError()

    def _load(self, location):
        """ Loads an ensemble """
        # for m in member_locations:
        #     m.append(self.model_factory(model_location, ...))
        # _save_ensemble_details
        raise NotImplementedError()

    def _on_model_end(self):
        """
        Default callback when a model finishes training.
        self._fit_loop_info contains information about the current step in the
        loop that trains all ensemble members.
        """
        pass

    def _on_model_start(self):
        """ Default callback when a model starts training """
        pass

    def fit(self):
        """ Train all Ensemble members """
        logging.info("=== Training Ensemble ===")
        start_time = time.perf_counter()
        cumulative_results = []
        for i, model in enumerate(self._members()):
            logging.info("\n\n\n=== Starting ensemble round %d / %d ===" % (i + 1, self.size))
            self._fit_loop_info = {
                'current_step': i,
                'current_model': model,
            }
            self._on_model_start()
            if self.wandb:
                run = wandb.init(project=self.wandb["project"], reinit=True,
                            config={"type": "ensemble", "params": self.model_params.__dict__},
                            group=self.wandb["group"],
                            name=f"member-{i}")
            self._fit_call(model)
            if self.wandb:
                run.finish()
            self._on_model_end()
            cumulative_results.append(self.evaluate(self.data.get_testing_handle()))
        end_time = time.perf_counter()
        m_summary = pd.DataFrame([m.test_metrics for m in self.members])
        if self.aggregator.is_fittable:
            #TODO: fit aggregator
            raise NotImplementedError()
        metrics = {'ensemble': self.evaluate(self.data.get_testing_handle()),
                'members': m_summary,
                'round_cumulative': cumulative_results,
                'time': end_time - start_time
        }
        if self.distil:
            logging.info("Training distilled model")
            if self.wandb:
                run = wandb.init(project=self.wandb["project"], reinit=True,
                            config={"type": "ensemble", "params": self.model_params.__dict__},
                            group=self.wandb["group"],
                            name=f"distilled-model")
            distilled_data = self.data.distillation_dataset(self)
            self.distilled_model = self.model_factory(params=self.model_params)
            self.distilled_model.fit(distilled_data, log_wandb=self.wandb)
            if self.wandb:
                run.finish()
            metrics['distilled_model'] = self.distilled_model.test_metrics
        return metrics
    
    def _fit_call(self, model):
        """ Wrapper for calling the model fitting """
        model.fit(self.data, log_wandb=self.wandb)

    def raw_predict_proba(self, X):
        """ Returns all the predictions from all Ensemble members """
        return np.array([m.predict_proba(X) for m in self.members[:self._fit_loop_info['current_step'] + 1]])

    def predict_proba(self, X):
        """ Return predicted soft probability outputs for the aggregate """
        return self.aggregator(self.raw_predict_proba(X), weights=self.model_weights)

    def predict_classes(self, X):
        """ Aggregated argmax """
        return np.argmax(self.predict_proba(X), axis = 1)

    def predict(self, X):
        """ Aggregated class values """
        return self.predict_proba(X)

    def evaluate(self, test_data=None):
        """ Evaluate model on some test data """
        #TODO: update for different data formats
        test_data = test_data or self.data.data['test']
        return tp.metrics.evaluate(self, test_data, self.members[0] if self.adversarial_testing else None)


### Ensemble Types / Class Templates ###

class Simple(EnsembleMethod):
    """
    A simple Ensemble - repeat the training N times and aggregate the results
    """
    def _initialise_members(self):
        self.members = [self.model_factory(params=self.model_params) for _ in range(self.size)]


class DynamicMembers(EnsembleMethod):
    """
    An Ensemble where the members are generated dynamically at each round
    """
    def _initialise_members(self):
        pass

    def _members(self):
        """ Must be implemented by the inheriting class """
        raise NotImplementedError()


### Actual Ensemble Methods ###

class Single(Simple):
    """
    A single model - run only once
    """
    def __init__(self, **kwargs):
        super().__init__(size=1, **kwargs)


class Bagging(Simple):
    """
    Bagging - TODO: documentation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data.resample()


#TODO: finish MA - sample counts, h, r
class AdaBoost(Simple):
    """
    Adaboost - TODO: finish documentation
    Currently supports two variants:
     - M1
     - MA (http://www.jmlr.org/papers/volume6/eibl05a/eibl05a.pdf)
    """
    def __init__(self, variant='M1', **kwargs):
        super().__init__(aggregator='averaging', **kwargs)
        self.variant = variant
        self.sample_weights = np.ones(self.data.size['train']) / float(self.data.size['train'])
        self.model_weights = np.ones(self.size)
        self.data = self.data.resample()


    def _on_model_end(self):
        """ Default callback when a model finishes training """
        model = self._fit_loop_info['current_model']
        y_true = []
        y_pred_p = []
        for (x, y_true_batch) in self.data.get_training_handle(resample=False): #TODO: non-resampled handle here
            y_true.append(np.argmax(y_true_batch, axis=1))
            y_pred_p.append(model.predict_proba(x))
        y_true = np.concatenate(y_true)
        y_pred_p = np.concatenate(y_pred_p)
        y_pred = np.argmax(y_pred_p, axis=1)
        if self.variant == 'MA': 
            y_pred_weights = np.max(y_pred_p, axis=1)
        else:
            y_pred_weights = 1.
        errors = (y_true == y_pred).astype('int32')
        e = np.sum(errors * self.sample_weights * y_pred_weights)
        if e > 0:
            alpha = .5 * math.log((1-e)/2) + math.log(self.data.n_classes-1)
            unnorm_weights = np.where(errors == 1,
                                      self.sample_weights * math.exp(alpha),
                                      self.sample_weights * math.exp(-alpha)
                                    )
            self.sample_weights = unnorm_weights / unnorm_weights.sum()
        self.data.set_weights(self.sample_weights)
        self.model_weights[self._fit_loop_info['current_step']] = alpha

    def _on_model_start(self):
        """ Default callback when a model starts training """
        self.data = self.data.resample()



class DIB(DynamicMembers):
    """
    Deep Incremental Boosting ()
    """
    def __init__(self, subsequent_epochs, insert_after, new_layers, variant='M1', **kwargs):
        super().__init__(aggregator='averaging', **kwargs)
        self.subsequent_epochs = subsequent_epochs
        self.new_layers = new_layers
        self.insert_after = insert_after
        self.variant = variant
        self.sample_weights = np.ones(self.data.size['train']) / float(self.data.size['train'])
        self.model_weights = np.ones(self.size)

    def _on_model_end(self):
        """ Default callback when a model finishes training """
        model = self._fit_loop_info['current_model']
        y_true = []
        y_pred_p = []
        for (x, y_true_batch) in self.data.get_training_handle(resample=False): #TODO: non-resampled handle here
            y_true.append(np.argmax(y_true_batch, axis=1))
            y_pred_p.append(model.predict_proba(x))
        y_true = np.concatenate(y_true)
        y_pred_p = np.concatenate(y_pred_p)
        y_pred = np.argmax(y_pred_p, axis=1)
        if self.variant == 'MA': 
            y_pred_weights = np.max(y_pred_p, axis=1)
        else:
            y_pred_weights = 1.
        errors = (y_true == y_pred).astype('int32')
        e = np.sum(errors * self.sample_weights * y_pred_weights)
        if e > 0:
            alpha = .5 * math.log((1-e)/2) + math.log(self.data.n_classes-1)
            unnorm_weights = np.where(errors == 1,
                                      self.sample_weights * math.exp(alpha),
                                      self.sample_weights * math.exp(-alpha)
                                    )
            self.sample_weights = unnorm_weights / unnorm_weights.sum()
        self.data.set_weights(self.sample_weights)
        self.model_weights[self._fit_loop_info['current_step']] = alpha

    def _on_model_start(self):
        """ Default callback when a model starts training """
        self.data = self.data.resample()

    def _members(self):
        """ Generator that creates new members on the fly by making the previous member bigger """
        for _ in range(self.size):
            new_member = self.model_factory(params=self.model_params)
            if self.members:
                new_member.inject_layers(self.new_layers, self.insert_after)
                new_member.copy_weights(self.members[-1])
            self.members.append(new_member)
            yield new_member

    def _fit_call(self, model):
        """ Wrapper for calling the model fitting """
        if self._fit_loop_info['current_step'] > 0:
            model.fit(self.data, self.subsequent_epochs, log_wandb=self.wandb)
        else:
            model.fit(self.data, log_wandb=self.wandb)