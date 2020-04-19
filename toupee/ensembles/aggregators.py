#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy as np
import toupee as tp

#TODO: Voting
#TODO: Stacking

class Aggregator:
    """
    Base class for all aggregating methods
    """
    def __init__(self):
        pass

    def __call__(self, Y, weights=None):
        raise NotImplementedError()

    def fit(self, X, y):
        pass


class Averaging(Aggregator):
    """
    Take an ensemble and produce the average, takes optional weights
    """

    def __init__(self):
        self.is_fittable = False

    def __call__(self, Y, weights=None):
        """ Calling interface to aggregate by average with optional weights """
        proba = np.array(Y)
        size = Y.shape[0]
        weights = weights or [1. / float(size) for _ in range(size)]
        return np.sum([proba[i] * weights[i] for i in range(size)], axis=0)


class MajorityVoting(Aggregator):
    """
    Use the argmax of an ensemble to get the votes and output the hard voting result
    """

    def __init__(self):
        self.is_fittable = False

    def __call__(self, Y, weights=None):
        """ Calling interface to aggregate by vote with optional weights """
        hard_Y = tp.data.one_hot_numpy(np.argmax(np.array(Y)))
        return Averaging.__call__(self, hard_Y, weights)
