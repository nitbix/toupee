#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import gc
import os
import sys
import time
import copy
import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.misc import imsave

import data
from data import Resampler, Transformer, sharedX
import update_rules
import layers
import config 
import cost_functions

class MLP(object):
    """
    Multi-Layer Perceptron (or any other kind of ANN if the layers exist)
    """

    def __init__(self, params, rng, input, index, x, y, pretraining_set = None):
        """
        Initialize the parameters for the multilayer perceptron
        """

        self.hiddenLayers = []
        self.chain_n_in = params.n_in
        self.chain_input_shape = None
        self.chain_in = input
        self.prev_dim = None
        self.params = params
        self.rng = rng
        layer_number = 0
        self.x = x
        self.y = y
        self.index = index

        def make_layer(layer_type,desc):
            if(layer_type == 'flat'):
                n_this,drop_this,name_this,activation_this = desc
                l = layers.FlatLayer(rng=rng, inputs=self.chain_in.flatten(ndim=2),
                                n_in=numpy.prod(self.chain_n_in), n_out=numpy.prod(n_this),
                                activation=activation_this,dropout_rate=drop_this,
                                layer_name=name_this)
                self.chain_n_in = n_this
                l.output_shape = self.chain_n_in
                self.chain_in=l.output
                return l
            elif(layer_type == 'conv'):
                input_shape,filter_shape,pool_size,drop_this,name_this,activation_this,pooling = desc
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
                l = layers.ConvolutionalLayer(rng=rng,
                                       inputs=self.chain_in, 
                                       input_shape=input_shape, 
                                       filter_shape=filter_shape,
                                       pool_size=pool_size,
                                       activation=activation_this,
                                       dropout_rate=drop_this,
                                       layer_name = name_this,
                                       pooling = pooling)
                prev_map_number,dim_x,dim_y = self.prev_dim
                curr_map_number = filter_shape[0]
                output_dim_x = (dim_x - filter_shape[2] + 1) / pool_size[0]
                output_dim_y = (dim_y - filter_shape[3] + 1) / pool_size[1]
                self.chain_n_in = (curr_map_number,output_dim_x,output_dim_y)
                l.output_shape = self.chain_n_in
                self.prev_dim = (curr_map_number,output_dim_x,output_dim_y)
                self.chain_in = l.output
                self.chain_input_shape = [self.chain_input_shape[0],
                        curr_map_number,
                        output_dim_x,
                        output_dim_y]
                return l

        for layer_type,desc in params.n_hidden:
            l = make_layer(layer_type,desc)
            self.hiddenLayers.append(l)

            def pretrain(pretraining_set,supervised = False, reverse = False):
                self.backup_first = None
                if reverse:
                    pretraining_set_x, pretraining_set_y = pretraining_set
                    pretraining_set_y = sharedX(
                            data.one_hot(pretraining_set_y.eval()))
                    x_pretraining = self.x
                    y_pretraining = T.matrix('y_pretraining')
                    reversedLayers = []
                    self.chain_in_back = self.chain_in
                    self.chain_n_in_back = self.chain_n_in
                    self.chain_in = y_pretraining
                    self.chain_n_in = self.params.n_out
                    backup = self.hiddenLayers
                    i = len(backup) - 1
                    for layer_type,desc in reversed(params.n_hidden[:len(backup)]):
                        l = make_layer(layer_type,desc)
                        l.W = backup[i].W
                        reversedLayers.append(l)
                        i -= 1
                    self.hiddenLayers = reversedLayers
                    self.make_top_layer(self.params.n_in, self.chain_in,
                            self.chain_n_in, rng, 'flat',
                            reversedLayers[0].activation)
                    train_model = self.train_function(index, pretraining_set_y,
                        pretraining_set_x, y_pretraining, x_pretraining)
                    ptxlen = pretraining_set_x.get_value(borrow=True).shape[0]
                    n_batches =  ptxlen / self.params.batch_size
                    for p in range(self.params.pretraining_passes):
                        print "... reverse training layer {0}, pass {1}".format(layer_number,p)
                        for minibatch_index in xrange(n_batches):
                            minibatch_avg_cost = train_model(minibatch_index,1)
                    self.hiddenLayers = backup
                    self.chain_in= self.chain_in_back 
                    self.chain_n_in= self.chain_n_in_back 
                    for i,l in enumerate(reversed(self.hiddenLayers)):
                        l.W = reversedLayers[i].W
                else:
                    if(supervised):
                        pretraining_set_x, pretraining_set_y = pretraining_set
                        x_pretraining = self.x
                        y_pretraining = self.y
                        self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,rng)
                    else:
                        pretraining_set_x = pretraining_set
                        pretraining_set_y = pretraining_set
                        ptylen = pretraining_set.get_value(borrow=True).shape[1]
                        x_pretraining = self.x
                        y_pretraining = T.matrix('y_pretraining')
                        self.make_top_layer(ptylen, self.chain_in, self.chain_n_in, rng,
                                'flat', self.hiddenLayers[0].activation)
                    train_model = self.train_function(index, pretraining_set_x,
                        pretraining_set_y, x_pretraining, y_pretraining)
                    ptxlen = pretraining_set_x.get_value(borrow=True).shape[0]
                    n_batches =  ptxlen / self.params.batch_size
                    for p in range(self.params.pretraining_passes):
                        print "... pretraining layer {0}, pass {1}".format(layer_number,p)
                        for minibatch_index in xrange(n_batches):
                            minibatch_avg_cost = train_model(minibatch_index,1)
                self.rejoin_layers(input)

            if pretraining_set is not None:
                mode = params.pretraining
                reverse = mode == 'reverse'
                #TODO: refactor into something decent
                if not isinstance(pretraining_set,tuple):
                    #unsupervised
                    pretrain(pretraining_set,False,reverse)
                elif len(pretraining_set) == 2:
                    #supervised
                    pretrain(pretraining_set,True,reverse)
                elif len(pretraining_set) == 3:
                    #both
                    pretrain(pretraining_set[2],False,reverse)
                    pretrain((pretraining_set[0],pretraining_set[1]),True)
            layer_number += 1
        self.rejoin_layers(input)
        self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,rng)

    def make_top_layer(self, n_out, chain_in, chain_n_in, rng,
            layer_type='softmax', 
            activation=None, name_this='temp_top'):
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
                layer_name='softmax')
            self.cost_function = self.params.cost_function
            self.p_y_given_x = self.outputLayer.p_y_given_x
            self.errors = self.outputLayer.errors
            self.y_pred = self.outputLayer.y_pred
        elif layer_type == 'flat':
            self.outputLayer = layers.FlatLayer(rng=rng,
                inputs=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in), n_out=n_out,
                activation=activation,dropout_rate=0,
                layer_name=name_this)
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

    def rejoin_layers(self,input):
        rt_chain_in = input
        for l in self.hiddenLayers:
            l.set_input(rt_chain_in)
            l.rejoin()
            rt_chain_in = l.output

    def eval_function(self, index, eval_set_x, eval_set_y, x, y):
        return theano.function(inputs=[index],
            outputs=self.errors(y),
            givens={
                x: eval_set_x[index * self.params.batch_size:(index + 1) *
                    self.params.batch_size],
                y: eval_set_y[index * self.params.batch_size:(index + 1) *
                    self.params.batch_size]})

    def compute(self, eval_set_x, x=None):
        x = self.x
        return theano.function(inputs=[],
            outputs=self.outputLayer.p_y_given_x,
            #outputs=self.outputLayer.y_pred,
            givens={ x: eval_set_x })

    def train_function(self, index, train_set_x, train_set_y, x, y):
        self.cost = self.cost_function(self.outputLayer,y) \
             + self.params.L1_reg * self.L1 \
             + self.params.L2_reg * self.L2_sqr
        self.gparams = []
        for param in self.opt_params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)
        previous_cost = T.scalar()
        updates = []
        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        dropout_rates = {}
        for layer in self.hiddenLayers:
            dropout_rates[layer.layer_name + '_W'] = layer.dropout_rate
        for param, gparam in zip(self.opt_params, self.gparams):
            if str(param) in dropout_rates.keys():
                include_prob = 1. - dropout_rates[str(param)]
            else:
                include_prob = 1.
            mask = theano_rng.binomial(p=include_prob,
                                       size=param.shape,dtype=param.dtype)    
            new_update = self.params.update_rule(param,
                    self.params.learning_rate, gparam, mask, updates,
                    self.cost,previous_cost)
            updates.append((param, new_update))
        return theano.function(inputs=[index,previous_cost],
                outputs=self.cost,
                on_unused_input='warn',
                updates=updates,
                givens={
                    x: train_set_x[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size],
                    y: train_set_y[index * self.params.batch_size:(index + 1) *
                        self.params.batch_size]
                })

    def make_models(self,dataset):
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = (None,None)
        print "..... eval"
        validate_model = self.eval_function(self.index, valid_set_x, valid_set_y,
                self.x, self.y)
        print "..... train"
        train_model = self.train_function(self.index, train_set_x, train_set_y,
                self.x, self.y)
        if len(dataset) > 2:
            test_set_x, test_set_y = dataset[2]
            print "..... test"
            test_model = self.eval_function(self.index, test_set_x,test_set_y,
                self.x, self.y)
        else:
            test_model = None
        return (train_model, validate_model, test_model)

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

def test_mlp(dataset, params, pretraining_set=None, x=None, y=None):
    state = {}
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = (None,None)

    print "training samples: {0}".format( train_set_x.get_value(borrow=True).shape[0])
    state['n_train_batches'] = train_set_x.get_value(borrow=True).shape[0] / params.batch_size
    state['n_valid_batches'] = valid_set_x.get_value(borrow=True).shape[0] / params.batch_size

    print '... building the model'

    index = T.lscalar()
    if x is None:
        x = T.matrix('x')
    if y is None:
        y = T.ivector('y')

    rng = numpy.random.RandomState(params.random_seed)

    state['classifier'] = MLP(params=params, rng=rng, input=x, index=index, x=x, y=y,
            pretraining_set=pretraining_set)

    if len(dataset) > 2:
        test_set_x, test_set_y = dataset[2]
        state['n_test_batches'] = test_set_x.get_value(borrow=True).shape[0] / params.batch_size

    print '... {0} training'.format(params.training_method)

    # early-stopping parameters
    state['patience'] = 10000  # look as this many examples regardless
    state['patience_increase'] = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(state['n_train_batches'], state['patience'] / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = time.clock()
    state['best_weights'] = None
    state['best_validation_loss'] = numpy.inf
    state['best_iter'] = 0
    state['test_score'] = 0.
    state['epoch'] = 0
    state['done_looping'] = False
    state['previous_minibatch_avg_cost'] = 1.
    state['dataset'] = dataset

    def run_hooks():
        updates = []
        params.learning_rate.epoch_hook(updates)
        one = sharedX(1.)
        f = theano.function(inputs=[],
                outputs=one,
                on_unused_input='warn',
                updates=updates,)
        f()

    def reset():
        updates = []
        params.learning_rate.reset(updates)
        one = sharedX(1.)
        f = theano.function(inputs=[],
                outputs=one,
                on_unused_input='warn',
                updates=updates,)
        f()

    def run_epoch():
        for minibatch_index in xrange(state['n_train_batches']):
            minibatch_avg_cost = state['train_model'](minibatch_index,
                    state['previous_minibatch_avg_cost'])
            iter = (state['epoch'] - 1) * state['n_train_batches'] + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [state['validate_model'](i) for i
                                     in xrange(state['n_valid_batches'])]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (state['epoch'], minibatch_index + 1, state['n_train_batches'],
                      this_validation_loss * 100.))
                if this_validation_loss < state['best_validation_loss']:
                    if this_validation_loss < state['best_validation_loss'] *  \
                           improvement_threshold:
                        state['patience'] = max(state['patience'],
                                iter * state['patience_increase'])
                    state['best_validation_loss'] = this_validation_loss
                    state['best_iter'] = iter
                    state['best_weights'] = state['classifier'].get_weights()
                    gc.collect()
                    # test it on the test set
                    if state['test_model'] is not None:
                        test_losses = [state['test_model'](i) for i
                                       in xrange(state['n_test_batches'])]
                        state['test_score'] = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (state['epoch'], minibatch_index + 1,
                                  state['n_train_batches'], state['test_score'] * 100.))
            state['previous_minibatch_avg_cost'] = minibatch_avg_cost
            if state['patience'] <= iter:
                    print('finished patience')
                    state['done_looping'] = True
                    break
        if params.save_images == True:
            for i in xrange(len(state['classifier'].hiddenLayers)):
                imsave('weights-layer{0}-iter{1}.png'.format(i,state['epoch']),
                        state['classifier'].hiddenLayers[i].W.get_value()
                      )
            imsave('weights-outputlayer-iter{0}.png'.format(state['epoch']),
                    state['classifier'].outputLayer.W.get_value()
                  )
#            for gparam in state['classifier'].gparams:
#              print gparam
#                imsave('gradient-{0}-iter{1}'.format(str(gparam),state['epoch']),
#                        gparam)
        run_hooks()

    if params.training_method == 'normal':
        state['best_weights'] = None
        state['best_validation_loss'] = numpy.inf
        state['best_iter'] = 0
        state['test_score'] = 0.
        state['epoch'] = 0
        state['done_looping'] = False
        print ".... generating models"
        state['train_model'], state['validate_model'], state['test_model'] = state['classifier'].make_models(dataset)
        reset()
        print ".... started"
        while (state['epoch'] < params.n_epochs) and (not state['done_looping']):
            state['epoch'] += 1
            run_epoch()
        state['classifier'].set_weights(state['best_weights'])

    elif params.training_method == 'greedy':
        all_layers = state['classifier'].hiddenLayers
        state['classifier'].hiddenLayers = []
        for l in xrange(len(all_layers)):
            reset()
            state['best_weights'] = None
            state['best_validation_loss'] = numpy.inf
            state['best_iter'] = 0
            state['test_score'] = 0.
            state['epoch'] = 0
            state['done_looping'] = False
            print "\n\ntraining {0} layers".format(l + 1)
            state['classifier'].hiddenLayers.append(all_layers[l])
            state['classifier'].rejoin_layers(x)
            state['classifier'].make_top_layer(
                    params.n_out,state['classifier'].hiddenLayers[l].output,
                    state['classifier'].hiddenLayers[l].output_shape,rng)
            print ".... generating models"
            state['train_model'], state['validate_model'], state['test_model'] = state['classifier'].make_models(dataset)
            print ".... started"
            while (state['epoch'] < params.n_epochs) and (not state['done_looping']):
                state['epoch'] += 1
                run_epoch()
            state['classifier'].set_weights(state['best_weights'])
    end_time = time.clock()
    if test_set_x is not None:
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (state['best_validation_loss'] * 100., state['best_iter'] + 1,
                  state['test_score'] * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        return state['classifier']
    else:
        print('Selection : Best validation score of {0} %'.format(
              state['best_validation_loss'] * 100.))
        return state['classifier']
