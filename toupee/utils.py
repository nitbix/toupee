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

class AppliedOnAllBatchesXY():

    def __init__(self, f, set_x, set_y, batch_size):
        self.f = f
        self.batch_size = batch_size
        self.original_size = set_x.shape.eval()[0]
        self.n_batches = int(floor(float(self.original_size) / self.batch_size))
        self.residue = self.original_size % self.batch_size
        self.set_x = set_x
        self.set_y = set_y
        if self.residue != 0:
            one = sharedX(1.)
            padding = self.batch_size - self.residue
            if padding > self.original_size:
                repeats = int(floor(padding / self.original_size)) + 1
                padding = padding % self.original_size
                self.f_pad = theano.function(
                    inputs=[],
                    outputs=one,
                    updates=[
                        ( set_x,
                          T.concatenate(
                              [T.repeat(set_x,repeats,axis=0),set_x[:padding]]
                          )
                        ),
                        ( set_y,
                          T.concatenate(
                              [T.repeat(set_y,repeats,axis=0),set_y[:padding]]
                          )
                        ),
                    ]
                )
            else:
                self.f_pad = theano.function(
                    inputs=[],
                    outputs=one,
                    updates=[
                        ( set_x,
                          T.concatenate([set_x,set_x[:padding]])),
                        ( set_y,
                          T.concatenate([set_y,set_y[:padding]]))
                    ]
                )
            self.f_unpad = theano.function(
                inputs=[],
                outputs=one,
                updates=[
                    (set_x, set_x[:self.original_size]),
                    (set_y, set_y[:self.original_size])
                ]
            )

    def clean_gpu(self):
        self.set_x.set_value([[]])
        self.set_y.set_value([])
        del self.f
        del self.f_pad
        del self.f_unpad

    def __call__(self):
        results = []
        for i in range(self.n_batches):
            results.append(self.f(i))
        if self.residue != 0:
            self.f_pad()
            results.append(self.f(self.n_batches))
            self.f_unpad()
        return T.concatenate(results)[:self.original_size]

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
    results = []
    original_size = set_x.shape.eval()[0]
    batches = int(floor(float(original_size) / batch_size))
    for i in range(batches):
        results.append(f(i))
    residue = original_size % batch_size
    if residue != 0:
        padding = batch_size - residue
        one = sharedX(1.)
        #the in-place padding and unpadding is ugly but makes life easier in
        #other places, so there is no need to alias/copy
        f_pad = theano.function(
            inputs=[],
            outputs=one,
            updates={ set_x : T.concatenate([set_x,set_x[:padding]]) }
        )
        f_unpad = theano.function(
            inputs=[],
            outputs=one,
            updates={ set_x : set_x[:original_size] }
        )
        f_pad()
        results.append(f(batches))
        f_unpad()
    return T.concatenate(results)[:original_size]

def apply_all_batches_xy(f, set_x, set_y, batch_size):
    results = []
    original_size = set_x.shape.eval()[0]
    batches = int(floor(float(original_size) / batch_size))
    for i in range(batches):
        results.append(f(i))
    residue = original_size % batch_size
    if residue != 0:
        padding = batch_size - residue
        one = sharedX(1.)
        #the in-place padding and unpadding is ugly but makes life easier in
        #other places, so there is no need to alias/copy
        f_pad = theano.function(
            inputs=[],
            outputs=one,
            updates=[
                (set_x , T.concatenate([set_x,set_x[:padding]])),
                (set_y , T.concatenate([set_y,set_y[:padding]]))
            ]
        )
        f_unpad = theano.function(
            inputs=[],
            outputs=one,
            updates=[
                (set_x, set_x[:original_size]),
                (set_y, set_y[:original_size])
            ]
        )
        f_pad()
        results.append(f(batches))
        f_unpad()
    return T.concatenate(results)[:original_size]

def set_slicer(x, i, set_x, output, batch_size, givens = {}):
    givens[x] = set_x[ i * batch_size : (i + 1) * batch_size ]
    return theano.function(
        inputs = [i],
        outputs = output,
        givens = givens 
    )

def set_slicer_xy(x, y, i, set_x, set_y, output, batch_size, givens = {}):
    givens[x] = set_x[ i * batch_size : (i + 1) * batch_size ]
    givens[y] = set_y[ i * batch_size : (i + 1) * batch_size ]
    return theano.function(
        on_unused_input='ignore',
        inputs = [i],
        outputs = output,
        givens = givens 
    )

def batched_computation(x, set_x, f, batch_size):
    index = T.lscalar()
    f = set_slicer(x, index, set_x, f, batch_size)
    return apply_all_batches(x, f, set_x, batch_size)
