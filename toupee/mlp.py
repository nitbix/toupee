#!/usr/bin/python
"""
Representation of a Multi-Layer Perceptron

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
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
from logistic_sgd import LogisticRegression
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
        prev_dim = None
        self.params = params
        self.rng = rng
        layer_number = 0

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
                    self.chain_input_shape = input_shape
                if prev_dim is None:
                    prev_dim = (input_shape[1],input_shape[2],input_shape[3])
                l = layers.ConvolutionalLayer(rng=rng,
                                       inputs=self.chain_in, 
                                       input_shape=input_shape, 
                                       filter_shape=filter_shape,
                                       pool_size=pool_size,
                                       activation=activation_this,
                                       dropout_rate=drop_this,
                                       layer_name = name_this,
                                       pooling = pooling)
                prev_map_number,dim_x,dim_y = prev_dim
                curr_map_number = filter_shape[0]
                output_dim_x = (dim_x - filter_shape[2] + 1) / pool_size[0]
                output_dim_y = (dim_y - filter_shape[3] + 1) / pool_size[1]
                self.chain_n_in = (curr_map_number,output_dim_x,output_dim_y)
                l.output_shape = self.chain_n_in
                prev_dim = (curr_map_number,output_dim_x,output_dim_y)
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
                    x_pretraining = x
                    y_pretraining = T.matrix('y_pretraining')
                    reversedLayers = []
                    self.chain_in_back = self.chain_in
                    self.chain_n_in_back = self.chain_n_in
                    self.chain_in = y_pretraining
                    self.chain_n_in = self.params.n_out
                    backup = self.hiddenLayers
                    for layer_type,desc in reversed(params.n_hidden):
                        l = make_layer(layer_type,desc)
                        reversedLayers.append(l)
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
                    for i,l in enumerate(self.hiddenLayers):
                        l.W = reversedLayers[i].W
                    rt_chain_in = input
                    for l in self.hiddenLayers:
                        l.inputs = rt_chain_in
                        l.rejoin()
                        rt_chain_in = l.output
                else:
                    if(supervised):
                        pretraining_set_x, pretraining_set_y = pretraining_set
                        x_pretraining = x
                        y_pretraining = y
                        self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,rng)
                    else:
                        pretraining_set_x = pretraining_set
                        pretraining_set_y = pretraining_set
                        ptylen = pretraining_set.get_value(borrow=True).shape[1]
                        x_pretraining = x
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

            if pretraining_set is not None:
                mode = params.pretraining
                reverse = mode == 'reverse'
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
        self.make_top_layer(self.params.n_out,self.chain_in,self.chain_n_in,rng)

    def make_top_layer(self, n_out, chain_in, chain_n_in, rng, layer_type='log', 
            activation=None, name_this='temp_top'):
        """
        Finalize the construction by making a top layer (either to use in
        pretraining or to use in the final version)
        """
        if layer_type == 'log':
            self.logRegressionLayer = LogisticRegression(
                input=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in),
                n_out=n_out)
            self.cost_function = self.params.cost_function
            self.p_y_given_x = self.logRegressionLayer.p_y_given_x
            self.errors = self.logRegressionLayer.errors
            self.y_pred = self.logRegressionLayer.y_pred
        elif layer_type == 'flat':
            self.logRegressionLayer = layers.FlatLayer(rng=rng,
                inputs=chain_in.flatten(ndim=2),
                n_in=numpy.prod(chain_n_in), n_out=n_out,
                activation=activation,dropout_rate=0,
                layer_name=name_this)
            self.cost_function = cost_functions.MSE()

        self.L1 = sum([abs(hiddenLayer.W).sum()
                    for hiddenLayer in self.hiddenLayers]) \
                + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = sum([(hiddenLayer.W ** 2).sum() for hiddenLayer in
                        self.hiddenLayers]) \
                    + (self.logRegressionLayer.W ** 2).sum()
        p = self.logRegressionLayer.params
        for hiddenLayer in self.hiddenLayers:
            p += hiddenLayer.params
        self.opt_params = p

    def eval_function(self,index,eval_set_x,eval_set_y,x,y):
        return theano.function(inputs=[index],
            outputs=self.errors(y),
            givens={
                x: eval_set_x[index * self.params.batch_size:(index + 1) * self.params.batch_size],
                y: eval_set_y[index * self.params.batch_size:(index + 1) * self.params.batch_size]})

    def train_function(self, index, train_set_x, train_set_y, x, y):
        self.cost = self.cost_function(self.logRegressionLayer,y) \
             + self.params.L1_reg * self.L1 \
             + self.params.L2_reg * self.L2_sqr
        gparams = []
        for param in self.opt_params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)
        previous_cost = T.lscalar()
        updates = []
        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        dropout_rates = {}
        for layer in self.hiddenLayers:
            dropout_rates[layer.layer_name + '_W'] = layer.dropout_rate
        for param, gparam in zip(self.opt_params, gparams):
            if str(param) in dropout_rates.keys():
                include_prob = 1. - dropout_rates[str(param)]
            else:
                include_prob = 1.
            mask = theano_rng.binomial(p=include_prob,
                                       size=param.shape,dtype=param.dtype)    
            new_update = self.params.update_rule(param, self.params.learning_rate, gparam, mask, updates,
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

def test_mlp(dataset, params, pretraining_set=None, x=None, y=None):
    global patience
    global classifier
    global best_iter
    global best_classifier
    global epoch
    global best_validation_loss
    global test_score
    global previous_minibatch_avg_cost
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = (None,None)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / params.batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / params.batch_size

    print '... building the model'

    index = T.lscalar()
    if x is None:
        x = T.matrix('x')
    if y is None:
        y = T.ivector('y')

    rng = numpy.random.RandomState(params.random_seed)

    classifier = MLP(params=params, rng=rng, input=x, index=index, x=x, y=y,
            pretraining_set=pretraining_set)

    if len(dataset) > 2:
        test_set_x, test_set_y = dataset[2]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / params.batch_size

    def make_models(classifier):
        validate_model = classifier.eval_function(index,valid_set_x,valid_set_y,x,y)
        train_model = classifier.train_function(index,train_set_x,train_set_y,x,y)
        if len(dataset) > 2:
            test_model = classifier.eval_function(index,test_set_x,test_set_y,x,y)
        else:
            test_model = None
        return (train_model, validate_model, test_model)

    print '... {0} training'.format(params.training_method)

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 20  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = time.clock()
    best_classifier = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    epoch = 0
    done_looping = False
    previous_minibatch_avg_cost = 1
    def run_hooks():
        updates = []
        params.learning_rate.epoch_hook(updates)
        one = sharedX(1.)
        f = theano.function(inputs=[],
                outputs=one,
                on_unused_input='warn',
                updates=updates,)
        f()

    def run_epoch():
        global patience
        global classifier
        global best_iter
        global best_classifier
        global epoch
        global best_validation_loss
        global test_score
        global previous_minibatch_avg_cost
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index,previous_minibatch_avg_cost)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_classifier = copy.copy(classifier)
                    # test it on the test set
                    if test_model is not None:
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
            if patience <= iter:
                    print('finished patience')
                    done_looping = True
                    break
        run_hooks()

    if params.training_method == 'normal':
        train_model, validate_model, test_model = make_models(classifier)
        while (epoch < params.n_epochs) and (not done_looping):
            epoch += 1
            run_epoch()

    elif params.training_method == 'greedy':
        all_layers = classifier.hiddenLayers
        for l in xrange(len(all_layers)):
            best_classifier = None
            best_validation_loss = numpy.inf
            best_iter = 0
            test_score = 0.
            epoch = 0
            done_looping = False
            print "training {0} layers\n".format(l + 1)
            classifier.hiddenLayers = all_layers[:l+1]
            classifier.make_top_layer(params.n_out,classifier.hiddenLayers[l].output,
                    classifier.hiddenLayers[l].output_shape,rng)
            train_model, validate_model, test_model = make_models(classifier)
            while (epoch < params.n_epochs) and (not done_looping):
                epoch = epoch + 1
                run_epoch()
            classifier = best_classifier
    end_time = time.clock()
    if test_set_x is not None:
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        return classifier
    else:
        print('Selection : Best validation score of {0} %'.format(
              best_validation_loss * 100.))
        return best_classifier

if __name__ == '__main__':
    #turn this on only if you want to do parameter search
    search_epochs = 40
    search = False

    params = config.load_parameters(sys.argv[1])
    dataset = data.load_data(params.dataset,
                              shared = True,
                              pickled = params.pickled)
    pretraining_set = data.make_pretraining_set(dataset,params.pretraining)
    if not search:
        mlp=test_mlp(dataset, params, pretraining_set = pretraining_set)
    else:
        params.n_epochs = search_epochs
        for eta_minus in [0.01,0.1,0.5,0.75,0.9]:
            params.update_rule.eta_minus = eta_minus
            for eta_plus in [1.001,1.01,1.1,1.2,1.5]:
                params.update_rule.eta_plus = eta_plus
                for min_delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                    params.update_rule.min_delta = min_delta
                    for max_delta in [50]:
                        print "PARAMS:"
                        print "ETA-: {0}".format(eta_minus)
                        print "ETA+: {0}".format(eta_plus)
                        print "MIN_DELTA: {0}".format(min_delta)
                        print "MAX_DELTA: {0}".format(max_delta)
                        params.update_rule.max_delta = max_delta
                        try:
                            mlp=test_mlp(dataset, params, pretraining_set = pretraining_set)
                        except KeyboardInterrupt:
                            print "skipping manually to next"
