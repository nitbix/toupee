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

import toupee as tp

#TODO: AdaBoost MA
#TODO: DIB
#TODO: BRN
#TODO: BaRN
#TODO: Snapshot
#TODO: ManyModels


class EnsembleMethod:
    """ Abstract representation of an Ensemble from which all other methods are derived """
    def __init__(self, data, size, model_params, aggregator, model_factory=tp.model.Model, saved_ensemble=None, **kwargs):
        self.data = data
        self.size = size
        self.model_params = model_params
        self.aggregator = tp.ensembles.get_aggregator(aggregator)
        self.members = []
        self.model_factory = model_factory
        self.model_weights = [1. / float(self.size) for _ in range(self.size)]
        if kwargs:
            logging.warning("Unknown ensemble parameters: %s" % kwargs)
        # contract: derived classes must set the members list
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
        for i, model in enumerate(self.members):
            logging.info("\n=== Model %d / %d ===" % (i + 1, len(self.members)))
            self._fit_loop_info = {
                'current_step': i,
                'current_model': model,
            }
            self._on_model_start()
            model.fit(self.data)
            self._on_model_end()
        end_time = time.perf_counter()
        m_summary = pd.DataFrame([m.test_metrics for m in self.members])
        if self.aggregator.is_fittable:
            #TODO: fit aggregator
            raise NotImplementedError()
        return {'ensemble': self.evaluate(self.data.get_testing_handle()),
                'members': m_summary,
                'time': end_time - start_time
        }
    
    def raw_predict_proba(self, X):
        """ Returns all the predictions from all Ensemble members """
        return np.array([m.predict_proba(X) for m in self.members])

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
        all_y_pred = []
        all_y_true = []
        for (x, y_true) in test_data:
            all_y_pred.append(self.predict_classes(x))
            all_y_true.append(np.argmax(y_true.numpy(), axis=1))
        y_pred = np.concatenate(all_y_pred)
        y_true = np.concatenate(all_y_true)
        return tp.utils.eval_scores(y_true, y_pred)


class Simple(EnsembleMethod):
    """
    A simple Ensemble - repeat the training N times and aggregate the results
    """
    def _initialise_members(self):
        self.members = [self.model_factory(params=self.model_params) for _ in range(self.size)]


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
        pass