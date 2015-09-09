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

rectifier = lambda x: T.maximum(0, x)
softsign = lambda x: x / (1 + abs(x))
tanh = T.tanh
