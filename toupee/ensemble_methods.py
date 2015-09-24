#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import sys
import numpy as np
import numpy.random
import theano
import theano.tensor as T
import yaml
from theano.sandbox.rng_mrg import MRG_RandomStreams

import mlp
from logistic_sgd import LogisticRegression
from data import Resampler, Transformer, sharedX, load_data, make_pretraining_set
from parameters import Parameters

floatX = theano.config.floatX

class AveragingRunner:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,x,y):
        self.members=members
        self.x = x
        self.y = y
        self.p_y_given_x = sum([m.p_y_given_x for m in self.members]) / len(members)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y), dtype=floatX, acc_dtype=floatX)


class MajorityVotingRunner:
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,x,y):
        self.members=members
        self.x = x
        self.y = y
        self.p_y_given_x = sum([T.eq(T.max(m.p_y_given_x),m.p_y_given_x)
            for m in self.members])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y), dtype=floatX, acc_dtype=floatX)


class WeightedAveraging:
    """
    Take an Ensemble and produce a weighted average, usually done in AdaBoost
    """

    def __init__(self,members,x,y):
        self.members=members
        self.x = x
        self.y = y
        #TODO: make this a Theano variable
        self.weights = [0 for x in members] + (1. / len(members))
        self.p_y_given_x = sum([T.eq(T.max(m.p_y_given_x),m.p_y_given_x)
            for m in self.members])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.mean(T.neq(self.y_pred, y), dtype=floatX, acc_dtype=floatX)

class StackingRunner:
    """
    Take an ensemble and produce the stacked output on a dataset
    """

    def __init__(self,members,x,y,train_set,valid_set,params):
        self.members=members
        train_set_x,train_set_y = train_set
        valid_set_x,valid_set_y = valid_set
        self.train_input_x = theano.function(inputs=[],
                on_unused_input='warn',
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.members],axis=1),
                givens={x:train_set_x})
        self.valid_input_x = theano.function(inputs=[],
                on_unused_input='warn',
                outputs=T.concatenate([m.p_y_given_x
                    for m in self.members],axis=1),
                givens={x:valid_set_x})
        print 'training stack head'
        self.head_x = T.concatenate([m.p_y_given_x
            for m in self.members],axis=1)
        dataset = ((sharedX(self.train_input_x(),borrow=True),train_set_y),
                   (sharedX(self.valid_input_x(),borrow=True),valid_set_y))
        pretraining_set = make_pretraining_set(dataset,params.pretraining)
        params.n_in = len(members) * params.main_params.n_out
        params.n_out = params.main_params.n_out
        self.stack_head = mlp.test_mlp(dataset, params,
                pretraining_set = pretraining_set, x = self.head_x, y = y)
        self.y_pred = self.stack_head.y_pred
        self.errors = self.stack_head.errors(y)


class DropStackingRunner:
    """
    Take an ensemble and produce the dropstacked output on a dataset
    """

    #TODO: should this be resampling a larger subset?
    def __init__(self,members,x,y,train_set,valid_set,params):
        self.rng = numpy.random.RandomState(params.random_seed)
        self.theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))
        self.members=members
        train_set_x,train_set_y = train_set
        valid_set_x,valid_set_y = valid_set
        self.train_input_x = theano.function(inputs=[],
                on_unused_input='warn',
                outputs=self.make_masked_input(params.dropstack_prob),
                givens={x:train_set_x})
        self.valid_input_x = theano.function(inputs=[],
                on_unused_input='warn',
                outputs=self.make_masked_input(params.dropstack_prob),
                givens={x:valid_set_x})
        print 'training stack head'
        self.head_x = T.concatenate([m.p_y_given_x
            for m in self.members],axis=1)
        dataset = ((sharedX(self.train_input_x(),borrow=True),train_set_y),
                   (sharedX(self.valid_input_x(),borrow=True),valid_set_y))
        pretraining_set = make_pretraining_set(dataset,params.pretraining)
        params.n_in = len(members) * params.main_params.n_out
        params.n_out = params.main_params.n_out
        self.stack_head = mlp.test_mlp(dataset, params,
                pretraining_set = pretraining_set, x = self.head_x, y = y)
        self.y_pred = self.stack_head.y_pred
        self.errors = self.stack_head.errors(y)

    def make_masked_input(self,prob):
        masked = []
        theano_rng = MRG_RandomStreams(max(self.members[0].rng.randint(2 ** 15), 1))
        mask = theano_rng.binomial(p=1-prob, size=(len(self.members),1))
        for i,m in enumerate(self.members):
            masked.append(m.p_y_given_x * mask[i,0])
        return T.concatenate(masked,axis=1)

class EnsembleMethod(yaml.YAMLObject):

    def create_aggregator(self,x,y,train_set,valid_set):
        raise NotImplementedException()

    def create_member(self,params,x,y,train_set,valid_set):
        raise NotImplementedException()

    def prepare(self, params, dataset):
        raise NotImplementedException()


class Bagging(EnsembleMethod):
    """
    Create a Bagging Runner from parameters
    """

    yaml_tag = u'!Bagging'
    def __init__(self,voting=False):
        self.voting = voting
        self.resampler = None

    def create_aggregator(self,params,members,x,y,train_set,valid_set):
        if 'voting' in self.__dict__ and self.voting:
            return MajorityVotingRunner(members,x,y)
        else:
            return AveragingRunner(members,x,y)

    def create_member(self,x,y):
        mlp_training_dataset = (self.resampler.make_new_train(self.params.resample_size),
                self.resampler.get_valid())
        pretraining_set = make_pretraining_set(mlp_training_dataset,self.params.pretraining)
        m = mlp.test_mlp(mlp_training_dataset, self.params,
                pretraining_set = pretraining_set, x=x, y=y)
        self.members.append(m)
        return m


    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.members = []


class AdaBoost(EnsembleMethod):
    """
    Create an AdaBoost Ensemble from parameters
    """

    yaml_tag = u'!AdaBoost'

    def create_aggregator(self,params,members,x,y,train_set,valid_set):
        #TODO: create weighted aggregator
        pass

    def create_member(self,x,y):
        mlp_training_dataset = (self.resampler.make_new_train(self.params.resample_size),
                self.resampler.get_valid())
        pretraining_set = make_pretraining_set(mlp_training_dataset,self.params.pretraining)
        m = mlp.test_mlp(mlp_training_dataset, self.params,
                pretraining_set = pretraining_set, x=x, y=y)
        self.members.append(m)
        return m

    def prepare(self, params, dataset):
        #TODO: create weighted resampler
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.members = []


class Stacking(EnsembleMethod):
    """
    Create a Stacking Runner from parameters
    """

    yaml_tag = u'!Stacking'

    def __init__(self,n_hidden,update_rule,n_epochs,batch_size,learning_rate,
            pretraining=None,pretraining_passes=1,training_method='normal',
            L1_reg=0.0,L2_reg=0.0):
        self.n_hidden = n_hidden
        self.update_rule = update_rule
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretraining = pretraining
        self.pretraining_passes = pretraining_passes
        self.training_method = training_method
        self.L1_reg = self.L1_reg
        self.L2_reg = self.L2_reg

    def create_aggregator(self,params,members,x,y,train_set,valid_set):
        self.main_params = self.params
        for p in self.params.__dict__:
            if p not in self.__dict__:
                self.__dict__[p] = self.params.__dict__[p]
        return StackingRunner(members,x,y,train_set,valid_set,
                Parameters(**self.__dict__))

    def create_member(self,x,y):
        mlp_training_dataset = (self.resampler.make_new_train(self.params.resample_size),
                self.resampler.get_valid())
        pretraining_set = make_pretraining_set(mlp_training_dataset,self.params.pretraining)
        m = mlp.test_mlp(mlp_training_dataset, self.params,
                pretraining_set = pretraining_set, x=x, y=y)
        self.members.append(m)
        return m

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.members = []


class DropStacking(Stacking):
    """
    Create a DropStacking Runner from parameters
    """

    yaml_tag = u'!DropStacking'

    def create_aggregator(self,params,members,x,y,train_set,valid_set):
        self.main_params = params
        for p in self.params.__dict__:
            if p not in self.__dict__:
                self.__dict__[p] = self.params.__dict__[p]
        return DropStackingRunner(members,x,y,train_set,valid_set,
                Parameters(**self.__dict__))

    def create_member(self,x,y):
        mlp_training_dataset = (self.resampler.make_new_train(self.params.resample_size),
                self.resampler.get_valid())
        pretraining_set = make_pretraining_set(mlp_training_dataset,self.params.pretraining)
        m = mlp.test_mlp(mlp_training_dataset, self.params,
                pretraining_set = pretraining_set, x=x, y=y)
        self.members.append(m)
        return m

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.members = []
