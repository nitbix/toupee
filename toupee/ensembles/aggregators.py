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

class Aggregator:
    """
    Base class for all aggregating methods
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()


class Averaging(Aggregator):
    """
    Take an ensemble and produce the average
    """

    def __init__(self):
        pass

    def __call__(self, Y):
        prob_arr = np.array(Y)
        a = np.sum(prob_arr,axis=0) / float(Y.shape[0])
        return a