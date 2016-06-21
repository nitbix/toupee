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
from math import floor

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

def apply_all_batches(x, f, set_x, batch_size):
    ys = []
    original_size = set_x.shape.eval()[0]
    batches = int(floor(float(original_size) / batch_size))
    for i in range(batches):
        ys.append(f(i))
    residue = original_size % batch_size
    if residue != 0:
        padding = batch_size - residue
        one = sharedX(1.)
        #the in-place padding and unpadding is ugly but makes life easier in
        #other places, so there is no need to alias/copy
        f_pad = theano.function(
            inputs=[],
            outputs=one,
            updates={
                set_x : 
                    T.concatenate([set_x,set_x[:padding]])
            }
        )
        f_unpad = theano.function(
            inputs=[],
            outputs=one,
            updates={
                set_x : 
                        set_x[:original_size]
            }
        )
        f_pad()
        ys.append(f(batches))
        f_unpad()
    y = T.concatenate(ys)[:original_size]
    return y

def set_slicer(x, i, set_x, output, batch_size):
    return theano.function(
        inputs = [i],
        outputs = output,
        givens = { x: set_x[ i * batch_size : (i + 1) * batch_size ] }
    )

def batched_computation(x, set_x, f, batch_size):
    index = T.lscalar()
    f = set_slicer(x, index, set_x, f, batch_size)
    return apply_all_batches(x, f, set_x, batch_size)
