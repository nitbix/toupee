#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T
import yaml

class Activation(yaml.YAMLObject):
    def __call__(self):
        raise NotImplementedError()

class TanH(Activation):
    yaml_tag = u'!TanH'
    def __call__(self,x):
        return T.tanh(x)
    def __str__(self):
        return 'TanH'

class ScaledTanH(Activation):
    yaml_tag = u'!ScaledTanH'
    def __call__(self,x):
        if 'A' not in self.__dict__:
            self.A = 1.
        if 'B' not in self.__dict__:
            self.B = 1.
        return self.A * T.tanh(self.B * x)
    def __str__(self):
        return 'ScaledTanH {{ A: {0} , B: {1} }}'.format(self.A,self.B)

class SoftSign(Activation):
    yaml_tag = u'!SoftSign'
    def __call__(self,x):
        return x / (1 + T.abs_(x))
    def __str__(self):
        return 'SoftSign'

class Sigmoid(Activation):
    yaml_tag = u'!Sigmoid'
    def __call__(self,x):
        return T.nnet.sigmoid(x)
    def __str__(self):
        return 'Sigmoid'

class ReLU(Activation):
    yaml_tag = u'!ReLU'
    def __call__(self,x):
        return T.nnet.relu(x)
    def __str__(self):
        return 'ReLU'

class BoundedReLU(Activation):
    yaml_tag = u'!BoundedReLU'
    def __call__(self,x):
        if 'rate' not in self.__dict__:
            self.rate = 0.01
        if 'ceiling' not in self.__dict__:
            self.ceiling = 6
        return T.min(T.nnet.relu(x,alpha=self.rate),self.ceiling)
    def __str__(self):
        return 'BoundedReLU'

class LeakyReLU(Activation):
    yaml_tag = u'!LeakyReLU'
    def __call__(self,x):
        if 'rate' not in self.__dict__:
            self.rate = 0.01
        return T.nnet.relu(x,alpha=self.rate)
    def __str__(self):
        return 'LeakyReLU {{ rate: {0} }}'.format(self.rate)
