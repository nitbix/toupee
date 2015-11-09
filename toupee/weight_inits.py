#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import yaml
from data import sharedX
import numpy
import theano
import theano.tensor as T

class WeightInit(yaml.YAMLObject):

    def __call__(self,rng,n_in,n_out,weight_name,activation):
        raise NotImplementedError()

class ZeroWeightInit(WeightInit):

    yaml_tag = u'!UniformWeightInit'

    def __call__(self,rng,n_in,n_out,weight_name,activation,shape=None):
        if shape is None:
            if n_out is None:
                shape = (n_in,)
            else:
                shape = (n_in, n_out)
        W_values = numpy.zeros(shape, dtype=theano.config.floatX)
        return theano.shared(value=W_values, name=weight_name, borrow=True)

class UniformWeightInit(WeightInit):

    yaml_tag = u'!UniformWeightInit'

    def __call__(self,rng,n_in,n_out,weight_name,activation,shape=None):
        if shape is None:
            if n_out is None:
                shape = (n_in,)
            else:
                shape = (n_in, n_out)
        W_values = numpy.asarray(rng.uniform(
                low=self.min_w,
                high=self.max_w,
                size=shape), dtype=theano.config.floatX)
        return theano.shared(value=W_values, name=weight_name, borrow=True)

class GlorotWeightInit(WeightInit):

    yaml_tag = u'!GlorotWeightInit'

    def __call__(self,rng,n_in,n_out,weight_name,activation,shape=None):
        if shape is None:
            if n_out is None:
                shape = (n_in,)
            else:
                shape = (n_in, n_out)
        W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=shape), dtype=theano.config.floatX)
        if type(activation) is 'toupee.activations.Sigmoid':
            W_values *= 4
        return theano.shared(value=W_values, name=weight_name, borrow=True)
