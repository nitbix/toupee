"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy as np
import theano
import theano.tensor as T
import yaml
import theano.tensor.extra_ops as TE
from data import sharedX

floatX = theano.config.floatX

def gauss(x, y, sigma=2.0):
    Z = 2 * np.pi * sigma**2
    return  1./Z * T.exp(-(x**2 + y**2) / (2. * sigma**2))

def gaussian_filter(kernel_shape):
    mid = T.floor(kernel_shape/ 2.)
    i = T.iscalar('i')
    j = T.iscalar('j')
    x,update = theano.scan(fn=lambda i : gauss((i / kernel_shape)-mid,
                                               (i % kernel_shape)-mid),
                sequences = [T.arange(kernel_shape ** 2, dtype=floatX)]
            )
    x = x.reshape((kernel_shape,kernel_shape))
    return x / x.sum()
