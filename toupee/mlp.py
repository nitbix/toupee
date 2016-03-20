#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import gc
import os
import sys
import time
import copy
import numpy
import scipy
import math
from pymongo import MongoClient
import json

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_from_host
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.misc import imsave

import data
from data import Resampler, Transformer, sharedX
import update_rules
import layers
import config 
import cost_functions
import activations
import common

floatX = theano.config.floatX

class TrainingState:
    """
    Helps track the state of the current training.
    """
    
    def __init__(self,classifier):
        self.reset()
        self.classifier = classifier
        self.best_valid_loss = numpy.inf

    def reset(self):
        self.done_looping = False
        self.best_weights = None
        self.best_iter = 0
        self.best_epoch = 0
        self.test_score = None
        self.epoch = 0
        self.n_batches = {}
        self.previous_minibatch_avg_cost = 1.

    def pre_iter(self):
        self.best_weights = None
        self.best_valid_loss = numpy.inf
        self.best_iter = 0
        self.best_epoch = 0
        self.test_score = 0.
        self.epoch = 0

    def set_models(self,models):
        self.train_f, self.train_error_f, self.valid_error_f, self.test_error_f = models


class MLP(object):
    """
    Multi-Layer Perceptron (or any other kind of ANN if the layers exist)
    """

    def __init__(self, params, rng, theano_rng, input, index, x, y, pretraining_set = None,
            continuation = None):
        """
        Initialize the parameters for the multilayer perceptron and create the
        network.
        """

        self.hiddenLayers = []
        self.layer_masks = {}
        self.layer_updates = {}
        self.chain_n_in = params.n_in
        self.chain_input_shape = None
        self.chain_in = input
        self.input = input
        self.prev_dim = None
        self.params = params
        self.rng = rng
        self.theano_rng = theano_rng
        layer_number = 0
        self.x = x
        self.y = y
        self.index = index
        self.reset_hooks(TrainingState(self))
        self.trainflag = T.scalar('trainflag')

        for i,(layer_type,desc) in enumerate(params.n_hidden):
            if continuation is not None:
                W = continuation['W'][i]
                b = continuation['b'][i]
            else:
                W = None
                b = None

            l = self.make_layer(layer_type,desc,W,b,i)
            self.hiddenLayers.append(l)

            modes = params.pretraining
            if pretraining_set is not None and modes is not None:
                for mode in modes.split(','):
                    self.inline_pretrain(pretraining_set,mode)
            layer_number += 1

        self.rejoin_layers(input)

        if pretraining_set is not None and modes is not None:
            for mode in modes.split(','):
                self.pretrain(pretraining_set,mode)
        if continuation is not None:
            W = continuation['outW']
            b = continuation['outb']
        else:
            W = None
            b = None
        self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,rng,
                layer_type=params.output_layer,W=W,b=b)


    def pretrain(self,pretraining_set,mode='unsupervised'):
        self.reset_hooks(TrainingState(self))
        if self.params.pretraining_noise is not None:
            pretraining_set[0] = data.corrupt(
                    self.params.pretraining_noise,pretraining_set[0])
        self.backup_first = None

        def reverse_epoch(pinned_layer=None):
            pretraining_set_x, pretraining_set_y = pretraining_set
            pretraining_set_y = sharedX(data.one_hot(pretraining_set_y.eval()))
            x_pretraining = self.x
            y_pretraining = T.matrix('y_pretraining')
            reversedLayers = []
            rev_chain_in = y_pretraining
            rev_chain_n_in = self.params.n_out
            backup = copy.copy(self.hiddenLayers)
            i = len(backup) - 1
            for layer_type,desc in reversed(self.params.n_hidden[:len(backup)]):
                if layer_type == 'conv':
                    print "Warning: reverse pretraining a convolutional layer has no effect"
                    reversedLayers.append(None)
                else:
                    l = self.make_layer(layer_type,desc,i)
                    l.W = backup[i].W
                    reversedLayers.append(l)
                    rev_chain_in = l.output
                    rev_chain_n_in = desc[0]
                    i -= 1
            if pinned_layer is not None and reversedLayers[pinned_layer] is not None:
                for i in range(0,len(reversedLayers) -1):
                    if reversedLayers[i] is not None:
                        reversedLayers[i].write_enable = 0
                reversedLayers[pinned_layer].write_enable = 1
            self.hiddenLayers = [x for x in reversedLayers if x is not None]
            self.make_top_layer(self.params.n_in, rev_chain_in,
                    rev_chain_n_in, self.rng, 'flat', activations.TanH())
            train_f = self.train_function(self.index, pretraining_set_y,
                pretraining_set_x, y_pretraining, x_pretraining,
                self.params.pretrain_update_rule,
                self.params.pretrain_learning_rate)
            ptxlen = pretraining_set_x.get_value(borrow=True).shape[0]
            n_batches = ptxlen / self.params.batch_size
            for p in range(self.params.pretraining_passes):
                print ".... pass {0}".format(p)
                for minibatch_index in xrange(n_batches):
                    minibatch_avg_cost = train_f(minibatch_index,1)
            self.hiddenLayers = backup
            for i,l in enumerate(reversed(self.hiddenLayers)):
                if reversedLayers[i] is not None:
                    l.W = reversedLayers[i].W
            del backup

        if mode == 'reverse':
            #greedy, one layer at a time
            for current_layer in xrange(0,len(self.hiddenLayers)):
                print "... reverse pretraining layer {0}".format(current_layer)
                reverse_epoch(current_layer)
            for i in range(0,len(self.hiddenLayers) -1):
                self.hiddenLayers[i].write_enable = 1
        elif mode == 'reverse-together':
            print "... reverse pretraining all layers"
            reverse_epoch()
        elif mode in ['supervised', 'supervised-together',
                'unsupervised', 'unsupervised-together']:
            #these happen inline
            return
        else:
            raise Exception("Unknown pretraining mode: %s" % mode)
        for l in self.hiddenLayers:
            l.write_enable = 1
        self.rejoin_layers(self.input)

    def inline_pretrain(self,pretraining_set,mode='unsupervised'):
        self.reset_hooks(TrainingState(self))
        if self.params.pretraining_noise is not None:
            pretraining_set[0] = data.corrupt(
                    self.params.pretraining_noise,pretraining_set[0])
        self.backup_first = None
        #these all lock the previous layers
        if mode in ['supervised', 'unsupervised']:
            for i in range(0,len(self.hiddenLayers) -1):
                self.hiddenLayers[i].write_enable = 0
            self.rejoin_layers(self.input)
        if mode in ['reverse', 'reverse-together']:
            #these happen at then end
            return
        elif mode in ['supervised', 'supervised-together']:
            pretraining_set_x, pretraining_set_y = pretraining_set
            x_pretraining = self.x
            y_pretraining = self.y
            self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,self.rng)
        elif mode in ['unsupervised', 'unsupervised-together']:
            pretraining_set_x = pretraining_set[0]
            pretraining_set_y = pretraining_set[0]
            ptylen = pretraining_set[0].get_value(borrow=True).shape[1]
            x_pretraining = self.x
            y_pretraining = T.matrix('y_pretraining')
            self.make_top_layer(ptylen, self.chain_in, self.chain_n_in, self.rng,
                    'flat', activations.TanH())
        else:
            raise Exception("Unknown pretraining mode: %s" % mode)
        if mode in ['supervised', 'unsupervised', 'unsupervised-together']:
            train_f = self.train_function(self.index, pretraining_set_x,
                pretraining_set_y, x_pretraining, y_pretraining)
            ptxlen = pretraining_set_x.get_value(borrow=True).shape[0]
            n_batches =  ptxlen / self.params.batch_size
            for p in range(self.params.pretraining_passes):
                print "... {0} pretraining layer {1}, pass {2}".format(mode,layer_number,p)
                for minibatch_index in xrange(n_batches):
                    minibatch_avg_cost = train_f(minibatch_index,1)
        for l in self.hiddenLayers:
            l.write_enable = 1
        self.rejoin_layers(self.input)

    def get_channels(self,i):
        """Try to guess the number of channels in the input from information
        in params or the input shape itself if it is not flat"""
        if i == 0:
            if 'channels' in self.params.__dict__:
                return self.params.channels
            elif self.params.RGB:
                return 3
        else:
            if len(self.chain_n_in) > 1:
                return self.chain_n_in[0]
        return 1

    def get_n_pixels(self,i):
        """Try to guess the square side of the input image from information
        in params or the input shape itself if it is not flat"""
        if i == 0:
            n_pixels_x = math.sqrt(numpy.prod(self.chain_n_in) / self.get_channels(i))
            n_pixels_y = n_pixels_x
        else:
            if len(self.chain_n_in) == 3:
                n_pixels_y = self.chain_n_in[1]
                n_pixels_x = self.chain_n_in[2]
            else:
                n_pixels_x = self.chain_n_in
                n_pixels_y = n_pixels_x
        return (n_pixels_y,n_pixels_x)

    def make_layer(self,layer_type,desc,W=None,b=None,i=0):
        if(layer_type == 'flat'):
            n_this,drop_this,name_this,activation_this,weight_init = desc
            l = layers.FlatLayer(rng=self.rng,
                                 inputs=self.chain_in.flatten(ndim=2),
                                 n_in=numpy.prod(self.chain_n_in),
                                 n_out=numpy.prod(n_this),
                                 activation=activation_this,
                                 dropout_rate=drop_this,
                                 layer_name=name_this,
                                 weight_init=weight_init,
                                 W=W,b=b)
            self.chain_n_in = n_this
            l.output_shape = self.chain_n_in
            self.chain_in=l.output
            return l
        elif(layer_type == 'LCN'):
            n_pixels_y,n_pixels_x = self.get_n_pixels(i)
            kernel_size,use_divisor = desc
            l = layers.LCN(
                        self.chain_in.flatten(ndim=2),
                        kernel_size,
                        n_pixels_x,
                        n_pixels_y,
                        self.get_channels(i),
                        use_divisor
                    )
            l.output_shape = self.chain_n_in
            self.chain_in=l.output
            return l
        elif(layer_type == 'elastic_transform'):
            n_pixels_y,n_pixels_x = self.get_n_pixels(i)
            l = layers.Elastic(
                        self.chain_in.flatten(ndim=2),
                        n_pixels_x,
                        n_pixels_y,
                        desc,
                        self.get_channels(i),
                        self.trainflag
                    )
            l.output_shape = self.chain_n_in
            self.chain_in=l.output
            return l
        elif(layer_type == 'dropout'):
            n_this,drop_this,name_this,activation_this,weight_init = desc
            l = layers.Dropout(rng=self.rng,
                                 inputs=self.chain_in.flatten(ndim=2),
                                 n_in=numpy.prod(self.chain_n_in),
                                 n_out=numpy.prod(self.chain_n_in),
                                 activation=activation_this,
                                 dropout_rate=drop_this,
                                 layer_name=name_this,
                                 )
            self.chain_n_in = n_this
            l.output_shape = self.chain_n_in
            self.chain_in=l.output
            return l
        elif(layer_type == 'conv'):
            if len(desc) == 8:
                #default border mode
                desc.append('valid')
            (input_shape,filter_shape,pool_size,drop_this,name_this,
                    activation_this,pooling,weight_init,border_mode) = desc
            if input_shape is None:
                if self.chain_input_shape is None:
                    raise Exception("must specify first input shape")
                input_shape = self.chain_input_shape
            else:
                if len(input_shape) == 3:
                    input_shape.insert(0,self.params.batch_size)
                self.chain_input_shape = input_shape
            if len(filter_shape) == 3:
                filter_shape.insert(1,input_shape[1])
            if self.prev_dim is None:
                self.prev_dim = (input_shape[1],input_shape[2],input_shape[3])
            l = layers.ConvolutionalLayer(rng=self.rng,
                                   inputs=self.chain_in, 
                                   input_shape=input_shape, 
                                   filter_shape=filter_shape,
                                   pool_size=pool_size,
                                   activation=activation_this,
                                   dropout_rate=drop_this,
                                   layer_name = name_this,
                                   pooling = pooling,
                                   border_mode = border_mode,
                                   weight_init = weight_init,
                                   W=W,b=b)
            prev_map_number,dim_x,dim_y = self.prev_dim
            curr_map_number = filter_shape[0]
            if border_mode == 'same':
                output_dim_x = dim_x / pool_size[0]
                output_dim_y = dim_y / pool_size[1]
            elif border_mode == 'valid':
                output_dim_x = (dim_x - filter_shape[2] + 1) / pool_size[0]
                output_dim_y = (dim_y - filter_shape[3] + 1) / pool_size[1]
            else:
                raise Exception('Invalid border mode: {0}'.format(border_mode))
            self.chain_n_in = (curr_map_number,output_dim_x,output_dim_y)
            l.output_shape = self.chain_n_in
            self.prev_dim = (curr_map_number,output_dim_x,output_dim_y)
            self.chain_in = l.output
            self.chain_input_shape = [self.chain_input_shape[0],
                    curr_map_number,
                    output_dim_x,
                    output_dim_y]
            return l

    def make_top_layer(self, n_out, chain_in, chain_n_in, rng,
            layer_type='softmax', activation=None, name_this='temp_top',
            W = None, b = None):
        """
        Finalize the construction by making a top layer (either to use in
        pretraining or to use in the final version)
        """
        if layer_type == 'softmax':
            self.outputLayer = layers.SoftMax(
                rng=rng,
                inputs=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in),
                n_out=n_out,
                activation=activation,
                dropout_rate=0,
                layer_name='softmax',
                W=W, b=b)
            self.cost_function = self.params.cost_function
            self.p_y_given_x = self.outputLayer.p_y_given_x
            self.errors = self.outputLayer.errors
            self.y_pred = self.outputLayer.y_pred
        if layer_type == 'logsoftmax':
            self.outputLayer = layers.LogSoftMax(
                rng=rng,
                inputs=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in),
                n_out=n_out,
                activation=activation,
                dropout_rate=0,
                layer_name='softmax',
                W=W,b=b)
            self.cost_function = self.params.cost_function
            self.p_y_given_x = self.outputLayer.p_y_given_x
            self.errors = self.outputLayer.errors
            self.y_pred = self.outputLayer.y_pred
        elif layer_type == 'flat':
            self.outputLayer = layers.FlatLayer(rng=rng,
                inputs=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in), n_out=n_out,
                activation=activation,dropout_rate=0,
                layer_name=name_this,
                W=W,b=b)
            self.cost_function = cost_functions.MSE()

        self.L1 = sum([abs(hiddenLayer.W).sum()
                    for hiddenLayer in self.hiddenLayers]) \
                + abs(self.outputLayer.W).sum()
        self.L2_sqr = sum([(hiddenLayer.W ** 2).sum() for hiddenLayer in
                        self.hiddenLayers]) \
                    + (self.outputLayer.W ** 2).sum()
        p = self.outputLayer.params
        for hiddenLayer in self.hiddenLayers:
            p += hiddenLayer.params
        self.opt_params = p

    def clear(self):
        del self.cost
        del self.hiddenLayers
        del self.outputLayer

    def rejoin_layers(self,input):
        rt_chain_in = input
        for l in self.hiddenLayers:
            l.set_input(rt_chain_in)
            l.rejoin()
            rt_chain_in = l.output

    def eval_function(self, index, eval_set_x, eval_set_y, x, y):
        return theano.function(
                inputs=[index],
                outputs=self.errors(y),
                on_unused_input='ignore',
                givens={
                    x: eval_set_x[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size],
                    y: eval_set_y[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size],
                    self.trainflag: numpy.float32(0.)
                }
               )

    def compute_single_batch(self, eval_set_x):
        x = self.x
        return theano.function(inputs=[],
            outputs=self.outputLayer.p_y_given_x,
            givens={ x: eval_set_x })

    def classify_single_batch(self, eval_set_x):
        x = self.x
        return theano.function(inputs=[],
            outputs=self.outputLayer.y_pred,
            givens={ x: eval_set_x })

    def apply_all_batches(self, f, eval_set_x):
        x = self.x
        ys = []
        original_size = eval_set_x.shape.eval()[0]
        batches = int(math.floor(float(original_size) / self.params.batch_size))
        print batches
        for i in range(batches):
            ys.append(f(i))
        residue = original_size % self.params.batch_size
        padding = self.params.batch_size - residue
        if residue != 0:
            one = sharedX(1.)
#the in-place padding and unpadding is ugly but makes life easier in other
#places, so that we don't need to alias/copy or do other inefficient things
            f_pad = theano.function(
                                inputs=[],
                                outputs=one,
                                updates={
                                    eval_set_x : 
                                        T.concatenate(
                                            [eval_set_x,eval_set_x[:padding]]
                                        )
                                }
                            )
            f_unpad = theano.function(
                                inputs=[],
                                outputs=one,
                                updates={
                                    eval_set_x : 
                                            eval_set_x[:original_size]
                                }
                            )
            f_pad()
            ys.append(f(batches))
            f_unpad()
        y = T.concatenate(ys)[:original_size]
        return y

    def compute(self, eval_set_x):
        index = T.lscalar()
        x = self.x
        f = theano.function(
                inputs=[index],
                outputs=self.outputLayer.p_y_given_x,
                givens={
                    x: eval_set_x[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size]
                })
        return self.apply_all_batches(f, eval_set_x)

    def classify(self, eval_set_x):
        index = T.lscalar()
        x = self.x
        f = theano.function(
                inputs=[index],
                outputs=self.outputLayer.y_pred,
                givens={
                    x: eval_set_x[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size]
                })
        return self.apply_all_batches(f, eval_set_x)

    def train_function(self, index, train_set_x, train_set_y, x, y,
            update_rule = None, learning_rate = None):
        if update_rule is None:
            update_rule = self.params.update_rule
        if learning_rate is None:
            learning_rate = self.params.learning_rate
        self.cost = self.cost_function(self.outputLayer,y)
        if self.params.L1_reg:
            self.cost += self.params.L1_reg * self.L1
        if self.params.L2_reg:
            self.cost += self.params.L2_reg * self.L2_sqr
        self.gparams = []
        for param in self.opt_params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)
        updates = []

        dropout_rates = {}
        write_enables = {}
        def unpack(layer):
            dropout_rates[layer.layer_name + '_W'] = layer.dropout_rate
            dropout_rates[layer.layer_name + '_b'] = 0.
            write_enables[layer.layer_name + '_W'] = layer.write_enable
            write_enables[layer.layer_name + '_b'] = layer.write_enable

        for layer in self.hiddenLayers:
            unpack(layer)
        unpack(self.outputLayer)

        self.previous_cost = T.scalar()
        for param, gparam in zip(self.opt_params, self.gparams):
            if str(param) in dropout_rates.keys():
                include_prob = 1. - dropout_rates[str(param)]
            else:
                include_prob = 1.
            if str(param) in write_enables.keys():
                we = write_enables[str(param)]
            else:
                raise Exception("missing write_enable for layer %s" % str(param))

            mask = data.mask(p=include_prob,shape=param.shape,dtype=param.dtype,
                    theano_rng=self.theano_rng)
            self.layer_masks[str(param)] = mask
            new_update = update_rule(param,
                    learning_rate, gparam, mask * we, updates,
                    self.cost,self.previous_cost)
            self.layer_updates[str(param)] = new_update
            for l in self.hiddenLayers:
                u = l.updates()
                if u is not None:
                    updates += u
            updates.append((param, new_update))
        rf = theano.function(
                inputs=[index,self.previous_cost],
                outputs=gpu_from_host(self.cost),
                on_unused_input='ignore',
                updates=updates,
                givens={
                    x: train_set_x[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size],
                    y: train_set_y[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size],
                    self.trainflag: sharedX(1.)
                })
        return rf

    def make_models(self, dataset):
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = (None,None)
        valid_error_f = self.eval_function(self.index, valid_set_x, valid_set_y,
                self.x, self.y)
        if self.params.online_transform is not None:
            train_f = None
            train_error_f = None
        else:
            train_f = self.train_function(self.index, train_set_x, train_set_y,
                self.x, self.y)
            train_error_f = self.eval_function(self.index, train_set_x,
                    train_set_y, self.x, self.y)
        if len(dataset) > 2:
            test_set_x, test_set_y = dataset[2]
            test_error_f = self.eval_function(self.index, test_set_x, test_set_y,
                self.x, self.y)
        else:
            test_error_f = None
        return (train_f, train_error_f, valid_error_f, test_error_f)

    def copy(self):
        """ Very expensive way of copying a network """
        newinstance = copy.copy(self)
        for i,l in enumerate(newinstance.hiddenLayers):
            l.copy_weights(self.hiddenLayers[i])
            l.rebuild()
        newinstance.outputLayer.copy_weights(self.outputLayer)
        newinstance.outputLayer.rebuild()
        return newinstance

    def get_weights(self):
        W = []
        b = []
        for l in self.hiddenLayers:
            W.append(l.W.get_value())
            b.append(l.b.get_value())
        outW = self.outputLayer.W.get_value()
        outb = self.outputLayer.b.get_value()
        return {'W': W, 'b': b, 'outW' : outW, 'outb': outb}

    def set_weights(self,weights):
        W = weights['W']
        b = weights['b']
        for i,l in enumerate(self.hiddenLayers):
            l.set_weights(W[i],b[i])
        self.outputLayer.set_weights(weights['outW'],weights['outb'])
        self.rejoin_layers(self.input)
        self.outputLayer.rejoin()

    def run_hooks(self):
        updates = []
        for hook in common.toupee_global_instance.epoch_hooks:
            hook(updates)
        one = sharedX(1.)
        f = theano.function(inputs=[],
                outputs=one,
                on_unused_input='warn',
                updates=updates,)
        f()

    def reset_hooks(self,state):
        state.done_looping = False
        state.pre_iter()
        updates = []
        for hook in common.toupee_global_instance.reset_hooks:
            hook(updates)
        one = sharedX(1.)
        f = theano.function(inputs=[],
                outputs=one,
                on_unused_input='warn',
                updates=updates)
        f()

def test_mlp(dataset, params, pretraining_set=None, x=None, y=None, index=None,
        continuation=None,return_results=False):
    results = common.Results(params)
    orig_train_set_x, orig_train_set_y = dataset[0]
    orig_valid_set_x, orig_valid_set_y = dataset[1]
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = (None,None)

    if params.online_transform is not None or params.join_train_and_valid:
        valid_set_x, valid_set_y  = data.shared_dataset(
                                        (numpy.concatenate([orig_train_set_x.eval({}),orig_valid_set_x.eval({})]),
                                         numpy.concatenate([orig_train_set_y.eval({}),orig_valid_set_y.eval({})])
                                        )
                                    )
        train_set_x, train_set_y = (valid_set_x,valid_set_y)
        dataset[0] = (train_set_x,train_set_y)
        dataset[1] = (valid_set_x,valid_set_y)
    if params.online_transform is not None:
        if 'channels' not in params.__dict__:
            if params.RGB:
                channels = 3
            else:
                channels = 1
        else:
            channels = params.channels
        gpu_transformer = data.GPUTransformer(valid_set_x,
                        x=int(math.sqrt(params.n_in / channels)),
                        y=int(math.sqrt(params.n_in / channels)),
                        channels=channels,
                        progress=False,
                        save=False,
                        opts=params.online_transform,
                        seed=params.random_seed)

    print "training samples: {0}".format( train_set_x.get_value(borrow=True).shape[0])

    if index is None:
        index = T.lscalar()
    if x is None:
        x = T.matrix('x')
    if y is None:
        y = T.ivector('y')

    rng = numpy.random.RandomState(params.random_seed)
    theano_rng = MRG_RandomStreams(params.random_seed)

    if continuation is None:
        classifier = MLP(params=params, rng=rng, theano_rng=theano_rng, input=x,
                index=index, x=x, y=y, pretraining_set=pretraining_set)
    else:
        classifier = MLP(params=params, rng=rng, theano_rng=theano_rng, input=x,
                index=index, x=x, y=y, pretraining_set=pretraining_set,
                continuation=continuation)

    state = TrainingState(classifier)
    state.n_batches['train'] = train_set_x.get_value(borrow=True).shape[0] / params.batch_size
    state.n_batches['valid'] = valid_set_x.get_value(borrow=True).shape[0] / params.batch_size
    if len(dataset) > 2:
        test_set_x, test_set_y = dataset[2]
        state.n_batches['test'] = test_set_x.get_value(borrow=True).shape[0] / params.batch_size

    print '... {0} training'.format(params.training_method)

    #TODO: Make these part of the YAML experiment description, after they get their own class
    # early-stopping parameters
    state.patience = 10000  # look as this many examples regardless
    state.patience_increase = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99   # a relative improvement of this much is
                                   # considered significant
    valid_frequency = min(state.n_batches['train'], state.patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the valid set; in this case we
                                  # check every epoch

    start_time = time.clock()

    def run_epoch(state,results):
        if params.online_transform is not None:
            train_set_x = gpu_transformer.get_data()
            state.train_f = state.classifier.train_function(
                    state.classifier.index,
                    train_set_x,
                    train_set_y,
                    state.classifier.x,
                    state.classifier.y)
            state.train_error_f = state.classifier.eval_function(
                    state.classifier.index,
                    train_set_x,
                    train_set_y,
                    state.classifier.x,
                    state.classifier.y)

        training_costs = []
        for minibatch_index in xrange(state.n_batches['train']):
            minibatch_avg_cost = state.train_f(minibatch_index,
                    state.previous_minibatch_avg_cost)
            training_costs.append(minibatch_avg_cost)
            iter = (state.epoch - 1) * state.n_batches['train'] + minibatch_index
            if (iter + 1) % valid_frequency == 0 \
                    or (minibatch_index + 1) == state.n_batches['train']:
                train_losses = [state.train_error_f(i) for i
                                     in xrange(state.n_batches['train'])]
                this_train_loss = numpy.mean(train_losses)
                valid_losses = [state.valid_error_f(i) for i
                                     in xrange(state.n_batches['valid'])]
                this_valid_loss = numpy.mean(valid_losses)
                print('epoch %i, minibatch %i/%i:' % 
                        (state.epoch, minibatch_index + 1,
                            state.n_batches['train'])
                     )
                print('  train err: %f %%' % (this_train_loss * 100.))
                print('  valid err: %f %%' % (this_valid_loss * 100.))
                test_loss = None
                if state.test_error_f is not None:
                    test_losses = [state.test_error_f(i) for i
                                   in xrange(state.n_batches['test'])]
                    test_loss = numpy.mean(test_losses)
                    print('  test  err: %f %%' % (test_loss * 100.))
                if this_valid_loss < state.best_valid_loss:
                    print('  current best')
                    if this_valid_loss < state.best_valid_loss *  \
                           improvement_threshold:
                        state.patience = max(state.patience, iter * state.patience_increase)
                    state.best_valid_loss = this_valid_loss
                    state.best_iter = iter
                    state.best_epoch = state.epoch
                    state.best_weights = state.classifier.get_weights()
                    state.test_score = test_loss
                    gc.collect()
                results.set_observation(this_train_loss,
                                        this_valid_loss,
                                        test_loss,
                                        numpy.array(minibatch_avg_cost))
            state.previous_minibatch_avg_cost = minibatch_avg_cost
            if state.patience <= iter:
                    print('finished patience')
                    state.done_looping = True
                    break
        if params.save_images or params.detailed_stats:
            e_x = numpy.asarray(dataset[0][0].eval())
            e_y = numpy.asarray(dataset[0][1].eval())
            padding_needed = params.batch_size - (len(e_x) % params.batch_size)
            padded_e_x = numpy.concatenate([
                    e_x,
                    e_x[:padding_needed]
                ])
            padded_e_y = numpy.concatenate([
                    e_y,
                    e_y[:padding_needed]
                ])
            e_xs = numpy.split(padded_e_x,len(padded_e_x) / params.batch_size)
            e_ys = numpy.split(padded_e_y,len(padded_e_y) / params.batch_size)
            assert len(e_xs) == len(e_ys)
            if params.save_images:
                for i in xrange(len(state.classifier.hiddenLayers)):
                    imsave('weights-layer{0}-iter{1}.png'.format(i,state.epoch),
                            state.classifier.hiddenLayers[i].W.get_value()
                          )
                imsave('weights-outputlayer-iter{0}.png'.format(state.epoch),
                        state.classifier.outputLayer.W.get_value()
                      )
            for param, gparam in zip(state.classifier.opt_params, state.classifier.gparams):
                gradients = []
                for i in xrange(0,len(e_xs)):
                    gradients.append(
                        numpy.asarray(gparam.eval({x: e_xs[i], y: e_ys[i]}))
                    )
                gradient = numpy.concatenate(gradients)
                p = numpy.asarray(param.eval())
                if params.save_images:
                    if len(gradient.shape) == 2:
                      imsave('gradient-{0}-iter{1}.png'.format(str(param),state.epoch),gradient * 255)
                if params.detailed_stats:
                    print "  {0} grad max: {1}".format(str(param),gradient.max())
                    print "  {0} max: {1}, min: {2}".format(str(param),p.max(),p.min())
                del gradient
                gc.collect()

            if params.detailed_stats:
                #for l,m in state.classifier.layer_updates.iteritems():
                #    print l
                #    print m.eval({x: e_x, y: e_y})
                #computed = state.classifier.classify(dataset[0][0])()
                #print "  output max: {0}, min: {1}, mean: {2}".format(computed.max(), computed.min(), computed.mean())
                print "  learning rate: {0}".format(params.learning_rate.get().get_value())
#                costs = []
#                for i in xrange(0,len(e_xs)):
#                    c = numpy.asarray(state.classifier.cost.eval({x: e_xs[i], y: e_ys[i]}))
#                    if len(c) > 0:
#                        costs.append(c)
#                cost = numpy.concatenate(costs)
#                if len(cost) > 0:
#                    print "  cost max: {0}, min: {1}, mean: {2}".format(cost.max(),cost.min(),cost.mean())
        state.classifier.run_hooks()
        if params.online_transform is not None:
            del train_set_x
            del state.train_f
            del state.train_error_f
            gc.collect()

    if params.training_method == 'normal':
        print ".... generating models"
        state.classifier.reset_hooks(state)
        state.set_models(state.classifier.make_models(dataset))
        print ".... started"
        while (state.epoch < params.n_epochs) and (not state.done_looping):
            epoch_start = time.clock()
            state.epoch += 1
            run_epoch(state,results)
            epoch_end = time.clock()
            print "t: {0}".format(epoch_end - epoch_start)
        if state.best_weights is not None:
            state.classifier.set_weights(state.best_weights)

    
    elif params.training_method == 'greedy':
        all_layers = state.classifier.hiddenLayers
        state.classifier.hiddenLayers = []
        for l in xrange(len(all_layers)):
            state.classifier.reset_hooks(state)
            print "\n\ntraining {0} layers".format(l + 1)
            state.classifier.hiddenLayers.append(all_layers[l])
            state.classifier.rejoin_layers(x)
            state.classifier.make_top_layer(
                    params.n_out,state.classifier.hiddenLayers[l].output,
                    state.classifier.hiddenLayers[l].output_shape,rng
                    )
            print ".... generating models"
            state.set_models(state.classifier.make_models(dataset))
            print ".... started"
            while (state.epoch < params.n_epochs) and (not state.done_looping):
                state.epoch += 1
                epoch_start = time.clock()
                run_epoch(state,results)
                epoch_end = time.clock()
                print "t: {0}".format(epoch_end - epoch_start)
            state.classifier.set_weights(state.best_weights)
    end_time = time.clock()
    if test_set_x is not None:
        print(('Optimization complete. Best valid score of %f %% '
               'obtained at iteration %i, epoch %i, with test performance %f %%') %
              (state.best_valid_loss * 100., state.best_iter + 1,
                  state.best_epoch, state.test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        results.set_final_observation(state.best_valid_loss * 100., state.test_score * 100., state.best_epoch)
    else:
        print('Selection : Best valid score of {0} %'.format(
              state.best_valid_loss * 100.))
    if params.online_transform is not None or params.join_train_and_valid:
        #restore original datasets that got messed about
        dataset[0] = orig_train_set_x, orig_train_set_y
        dataset[1] = orig_valid_set_x, orig_valid_set_y
        del test_set_x
        test_set_x = dataset[0][0]
        valid_set_x = dataset[1][0]
    cl = state.classifier
    del state
    gc.collect()
    if 'results_db' in params.__dict__ :
        if 'results_host' in params.__dict__:
            host = params.results_host
        else:
            host = None
        print "saving MLP results to {0}@{1}".format(params.results_db,host)
        conn = MongoClient(host=host)
        db = conn[params.results_db]
        if 'results_table' in params.__dict__: 
            table_name = params.results_table
        else:
            table_name = 'results'
        table = db[table_name]
        def serialize(o):
            if isinstance(o, numpy.float32):
                return float(o)
            else:
                try:
                    return numpy.asfarray(o).tolist()
                except:
                    if isinstance(o, object):
                        if 'tolist' in dir(o) and callable(getattr(o,'tolist')):
                            return o.tolist()
                        try:
                            return json.loads(json.dumps(o.__dict__,default=serialize))
                        except:
                            return str(o)
                    else:
                        raise Exception("don't know how to save {0}".format(type(o)))
        table.insert(json.loads(json.dumps(results.__dict__,default=serialize)))
    if return_results:
        return cl,results
    else:
        return cl
