#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy as np
import mlp
from data import Resampler, WeightedResampler
import common
import math
import keras
from keras.layers import Input, Convolution2D, merge
from keras.models import Model
from pprint import pprint
import copy

class Aggregator:
    """
    Base class for all aggregating methods
    """
    def __init__(self):
        pass

    def compute(self, set_x):
        raise NotImplementedException()

    def predict(self, set_x):
        raise NotImplementedException()

    def predict_classes(self, set_x):
        return np.argmax(self.predict(set_x), axis = 1)


class AveragingRunner(Aggregator):
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,params):
        self.params = params
        self.members = members

    def predict(self,data):
        prob = []
        for (m_yaml, m_weights) in self.members:
            m = keras.models.model_from_yaml(m_yaml)
            m.set_weights(m_weights)
            p = m.predict_proba(data, batch_size = self.params.batch_size)
            prob.append(p)
            out_shape = m.layers[-1].output_shape
        prob_arr = np.array(prob)
        a = np.sum(prob_arr,axis=0) / float(len(self.members))
        m = np.argmax(a,axis=1)
        return np.eye(out_shape[1])[m]


class MajorityVotingRunner(Aggregator):
    """
    Take an ensemble and produce the majority vote output on a dataset
    """

    def __init__(self,members,params):
        self.params = params
        self.members = members

    def predict(self,data):
        classifs = []
        for (m_yaml, m_weights) in self.members:
            m = keras.models.model_from_yaml(m_yaml)
            m.set_weights(m_weights)
            c = m.predict(data, batch_size = self.params.batch_size)
            classifs.append(c)
            out_shape = m.layers[-1].output_shape
        classifs_arr = np.array(classifs)
        a = np.sum(classifs_arr,axis=0) / float(len(self.members))
        m = np.argmax(a,axis=1)
        return np.eye(out_shape[1])[m]


class WeightedAveragingRunner(Aggregator):
    """
    Take an Ensemble and produce a weighted average, usually done in AdaBoost
    """


    def __init__(self,members,weights,params):
        self.params = params
        self.members = members
        self.weights = weights

    def predict(self,data):
        prob = []
        for i in range(len(self.members)):
            m_yaml, m_weights = self.members[i]
            m = keras.models.model_from_yaml(m_yaml)
            m.set_weights(m_weights)
            p = m.predict_proba(data, batch_size = self.params.batch_size)
            prob.append(p * self.weights[i])
            out_shape = m.layers[-1].output_shape
        prob_arr = np.array(prob) / np.sum(self.weights)
        a = np.sum(prob_arr,axis=0) / float(len(self.members))
        m = np.argmax(a,axis=1)
        return np.eye(out_shape[1])[m]


#class StackingRunner(Aggregator):
#    """
#    Take an ensemble and produce the stacked output on a dataset
#    """
#
#    def join_outputs(self, x, set_x_shared, batch_size, p=0.):
#        rng = numpy.random.RandomState()
#        set_x = set_x_shared.eval()
#        n_instances = set_x.shape[0]
#        n_batches = n_instances / batch_size
#        acc = []
#        for b in xrange(0,n_batches):
#            start = b * batch_size
#            end = (start + batch_size) % n_instances
#            curr_x = set_x[start:end]
#            curr_model_outs = numpy.concatenate([m.p_y_given_x.eval({x:curr_x})
#                                    for i,m in enumerate(self.members)], axis=1)
#            n_m = len(self.members)
#            mask = rng.binomial(1, 1.-p, (batch_size,n_m))
#            mask = numpy.repeat(mask, curr_model_outs.shape[1] / n_m, axis = 1)
#            acc.append(curr_model_outs * mask)
#        r = numpy.concatenate(acc)
#        return sharedX(r)
#
#    def __init__(self,members,x,y,train_set,valid_set,params):
#        self.params = params
#        self.members = members
#        train_set_x,train_set_y = train_set
#        valid_set_x,valid_set_y = valid_set
#        if 'dropstack_prob' not in params.__dict__:
#            p = 0.
#        else:
#            p = params.dropstack_prob
#        self.train_input_x = self.join_outputs(x, train_set_x, params.batch_size, p)
#        self.valid_input_x = self.join_outputs(x, valid_set_x, params.batch_size, p)
#        print 'training stack head'
#        self.head_x = T.concatenate([m.p_y_given_x
#                                    for m in self.members],axis=1)
#        dataset = ((self.train_input_x,train_set_y),
#                   (self.valid_input_x,valid_set_y))
#        pretraining_set = make_pretraining_set(dataset,params.pretraining)
#        params.n_in = len(members) * params.main_params.n_out
#        params.n_out = params.main_params.n_out
#        self.stack_head = mlp.test_mlp(dataset, params,
#                pretraining_set = pretraining_set, x = self.head_x, y = y)
#        self.y_pred = self.stack_head.y_pred
#        self.errors = self.stack_head.errors(y)
#
#
class EnsembleMethod(common.ConfiguredObject):

    def _default_value(self, param_name, value):
        if param_name not in self.__dict__:
            print "WARNING: setting default for: {0} to {1}".format(param_name, value) 
            self.__dict__[param_name] = value

    def create_aggregator(self,x,y,train_set,valid_set):
        raise NotImplementedException()

    def create_member(self):
        raise NotImplementedException()

    def prepare(self, params, dataset):
        raise NotImplementedException()

    def load_weights(self,weights,x,y,index):
        self.members = []
        self.weights = []
        raise "needs fixing"
        for w in weights:
            rng = numpy.random.RandomState(self.params.random_seed)
            m = mlp.MLP(params=self.params, rng=rng, input=x, index=index, x=x, y=y)
            m.set_weights(w)
            self.members.append(m)
        return self.members

    def serialize(self):
        return 'UnknownEnsemble'


class Bagging(EnsembleMethod):
    """
    Create a Bagging Runner from parameters
    """

    yaml_tag = u'!Bagging'
    def __init__(self,voting=False):
        self.voting = voting
        self.resampler = None
    
    def create_aggregator(self,params,members,train_set,valid_set):
        if 'voting' in self.__dict__ and self.voting:
            return MajorityVotingRunner(members,params)
        else:
            return AveragingRunner(members,params)

    def create_member(self):
        train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
        if self.member_number > 0 :
            resampled = [
                train_set,
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        else:
            train_weights = None
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        m = mlp.sequential_model(resampled, self.params,
                member_number = self.member_number)
        self.member_number += 1
        return (m.to_yaml(), m.get_weights())

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.member_number = 0

    def serialize(self):
        return 'Bagging'


class DIB(EnsembleMethod):
    """
    Create Deep Incremental Boosting Ensemble from parameters
    """

    #TODO: use sample weight
    yaml_tag = u'!DIB'

    def set_defaults(self):
        self._default_value('incremental_index', -1)
        self._default_value('use_sample_weights', False)

    def create_aggregator(self,params,members,train_set,valid_set):
            return WeightedAveragingRunner(members,self.alphas,params)

    def create_member(self):
        self.set_defaults()
        if self.member_number > 0 :
            train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
            resampled = [
                train_set,
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        else:
            train_weights = None
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        if self.member_number > 0:
            self.params.n_epochs = self.n_epochs_after_first
            if 'lr_after_first' in self.params.__dict__:
                self.params.optimizer['config']['lr'] = self.params.lr_after_first
        if not self.use_sample_weights:
            train_weights = None
        m = mlp.sequential_model(resampled, self.params,
            member_number = self.member_number, model_weights = self.weights,
            #the copy is because there is a bug in Keras that deletes names
            model_config = copy.deepcopy(self.model_config),
            sample_weight = train_weights)
        self.weights = [l.get_weights() for l in m.layers]
        injection_index = self.incremental_index + self.member_number * len(self.incremental_layers)
        if self.incremental_layers is not None:
            if injection_index == -1:
                injection_index = len(self.model_config)
            new_layers = []
            for i,l in enumerate(self.incremental_layers):
                l['config']['name'] = "DIB-incremental-{0}-{1}".format(
                    self.member_number, i)
                new_layers.append(l)
            new_model_config = self.model_config[:injection_index] + new_layers + self.model_config[injection_index:]
            self.model_config = copy.deepcopy(new_model_config)
            self.weights = self.weights[:injection_index]
        orig_train = self.resampler.get_train()
        errors = common.errors(m, orig_train[0], orig_train[1])
        e = np.sum((errors * self.D))
        alpha = .5 * math.log((1-e)/e)
        w = np.where(errors == 1,
            self.D * math.exp(alpha),
            self.D * math.exp(-alpha))
        self.D = w / w.sum()
        self.resampler.update_weights(self.D)
        self.alphas.append(alpha)
        self.member_number += 1
        m_yaml = m.to_yaml()
        m_weights = m.get_weights()
        del m
        return (m_yaml, m_weights)

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = WeightedResampler(dataset)
        self.D = self.resampler.weights
        self.weights = None
        self.member_number = 0
        self.alphas = []
        with open(params.model_file, 'r') as model_file:
            model_yaml = model_file.read()
        self.model_config = keras.models.model_from_yaml(model_yaml).get_config()

    def serialize(self):
        return 'DIB'


    def serialize(self):
        self.set_defaults()
        return """
DIB {{
    n_epochs_after_first: {0},
    incremental_index: {1},
    use_sample_weights: {2},
    incremental_layers: {3}
}}
        """.format(self.n_epochs_after_first,
                   self.use_sample_weights,
                   self.incremental_index,
                   self.incremental_layers)


class BRN(EnsembleMethod):
    """
    Create Residual Incremental Boosting Ensemble from parameters
    """

    #TODO: use sample weight
    yaml_tag = u'!BRN'

    def set_defaults(self):
        self._default_value('incremental_index', -1)
        self._default_value('use_sample_weights', False)
        self._default_value('resample', False)

    def create_aggregator(self,params,members,train_set,valid_set):
            return WeightedAveragingRunner(members,self.alphas,params)

    def _residual_block(self, injection_index, new_layers, m, member_number):
        #get output shape of last layer before injection from m
        if injection_index > 0:
            input_shape = m.layers[injection_index - 1].output_shape
        else:
            input_shape = m.input_shape
        #make input
        input_layer = Input(shape = input_shape[1:], name = "Input_BRN_{0}".format(member_number))
        #make real layers
        real_layers = input_layer
        for i,l in enumerate(new_layers):
            l['config']['name'] = "BRN-incremental-{0}-{1}".format(
                member_number, i)
            real_layers = keras.layers.deserialize(l)(real_layers)
        #make skip layer
        stride_width = input_shape[2] / real_layers._keras_shape[2]
        stride_height = input_shape[3] / real_layers._keras_shape[3]
        equal_channels = real_layers._keras_shape[1] == input_shape[1]
        shortcut = input_layer
        # 1 X 1 conv if shape is different. Else identity.
        if (stride_width > 1 or stride_height > 1 or not equal_channels) and stride_width > 0 and stride_height > 0:
            shortcut = Convolution2D(nb_filter=real_layers._keras_shape[1], nb_row=1, nb_col=1,
                                     subsample=(stride_width, stride_height),
                                     init="he_normal",
                                     border_mode="same",
                                     name="shortcut_BRN_{0}".format(member_number))(input_layer)

        #make merge
        merge_layer = merge([real_layers,shortcut], mode="sum", name = "merge_BRN_{0}".format(member_number))
        #make model
        model = Model(inputs=[input_layer],outputs=[merge_layer],
                name="Model_BRN_{0}".format(member_number))
        #make config
        return {"class_name": "Model", "config": model.get_config()}

    def create_member(self):
        self.set_defaults()
        if self.member_number > 0 :
            if self.resample:
                train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
                resampled = [
                    train_set,
                    self.resampler.get_valid(),
                    self.resampler.get_test()
                ]
            else:
                train_weights = self.D
                resampled = [
                    self.resampler.get_train(),
                    self.resampler.get_valid(),
                    self.resampler.get_test()
                ]
        else:
            train_weights = None
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        if self.member_number > 0:
            self.params.n_epochs = self.n_epochs_after_first
        if not self.use_sample_weights:
            train_weights = None
        m = mlp.sequential_model(resampled, self.params,
            member_number = self.member_number, model_weights = self.weights,
            #the copy is because there is a bug in Keras that deletes names
            model_config = copy.deepcopy(self.model_config),
            frozen_layers = self.frozen_layers,
            sample_weight = train_weights)
        self.weights = [l.get_weights() for l in m.layers]
        injection_index = self.incremental_index + self.member_number
        if self.incremental_layers is not None:
            if injection_index == -1:
                injection_index = len(self.model_config)
            new_layers = []
            for i,l in enumerate(self.incremental_layers):
                new_layers.append(copy.deepcopy(l))
            #make residual block
            new_block = self._residual_block(injection_index, new_layers, m,
                    self.member_number)
            new_model_config = self.model_config[:injection_index] + [new_block] + self.model_config[injection_index:]
            if self.freeze_old_layers:
                self.frozen_layers = range(0,injection_index)
            self.model_config = copy.deepcopy(new_model_config)
            self.weights = self.weights[:injection_index]
        orig_train = self.resampler.get_train()
        errors = common.errors(m, orig_train[0], orig_train[1])
        e = np.sum((errors * self.D))
        alpha = .5 * math.log((1-e)/e)
        w = np.where(errors == 1,
            self.D * math.exp(alpha),
            self.D * math.exp(-alpha))
        self.D = w / w.sum()
        self.resampler.update_weights(self.D)
        self.alphas.append(alpha)
        self.member_number += 1
        return (m.to_yaml(), m.get_weights())

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = WeightedResampler(dataset)
        self.D = self.resampler.weights
        self.weights = None
        self.member_number = 0
        self.alphas = []
        self.frozen_layers = []
        with open(params.model_file, 'r') as model_file:
            model_yaml = model_file.read()
        self.model_config = keras.models.model_from_yaml(model_yaml).get_config()

    def serialize(self):
        self.set_defaults()
        return """
BRN {{
    n_epochs_after_first: {0},
    incremental_index: {1},
    use_sample_weights: {2},
    incremental_layers: {3}
}}
        """.format(self.n_epochs_after_first,
                   self.use_sample_weights,
                   self.incremental_index,
                   self.incremental_layers)


class BARN(EnsembleMethod):
    """
    Create Residual Incremental Boosting Ensemble from parameters
    """
    #TODO: use sample weight

    yaml_tag = u'!BARN'

    def set_defaults(self):
        self._default_value('incremental_index', -1)
        self._default_value('use_sample_weights', False)

    def create_aggregator(self,params,members,train_set,valid_set):
            return AveragingRunner(members, params)

    def _residual_block(self, injection_index, new_layers, m, member_number):
        #get output shape of last layer before injection from m
        if injection_index > 0:
            input_shape = m.layers[injection_index - 1].output_shape
        else:
            input_shape = m.input_shape
        #make input
        input_layer = Input(shape = input_shape[1:], name = "Input_BARN_{0}".format(member_number))
        #make real layers
        real_layers = input_layer
        for i,l in enumerate(new_layers):
            l['config']['name'] = "BARN-incremental-{0}-{1}".format(
                member_number, i)
            real_layers = keras.layers.deserialize(l)(real_layers)
        #make skip layer
        stride_width = input_shape[2] / real_layers._keras_shape[2]
        stride_height = input_shape[3] / real_layers._keras_shape[3]
        equal_channels = real_layers._keras_shape[1] == input_shape[1]
        shortcut = input_layer
        # 1 X 1 conv if shape is different. Else identity.
        if (stride_width > 1 or stride_height > 1 or not equal_channels) and stride_width > 0 and stride_height > 0:
            shortcut = Convolution2D(nb_filter=real_layers._keras_shape[1], nb_row=1, nb_col=1,
                                     subsample=(stride_width, stride_height),
                                     init="he_normal",
                                     border_mode="same",
                                     name="shortcut_BARN_{0}".format(member_number))(input_layer)

        #make merge
        merge_layer = merge([real_layers,shortcut], mode="sum", name = "merge_BARN_{0}".format(member_number))
        #make model
        model = Model(input=input_layer,output=merge_layer,
                name="Model_BARN_{0}".format(member_number))
        #make config
        return {"class_name": "Model", "config": model.get_config()}

    def create_member(self):
        self.set_defaults()
        if self.member_number > 0 :
            train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
            resampled = [
                train_set,
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        else:
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        train_weights = None
        if self.member_number > 0:
            self.params.n_epochs = self.n_epochs_after_first
        m = mlp.sequential_model(resampled, self.params,
            member_number = self.member_number, model_weights = self.weights,
            #the copy is because there is a bug in Keras that deletes names
            model_config = copy.deepcopy(self.model_config),
            frozen_layers = self.frozen_layers)
        self.weights = [l.get_weights() for l in m.layers]
        injection_index = self.incremental_index + self.member_number
        if self.incremental_layers is not None:
            if injection_index == -1:
                injection_index = len(self.model_config)
            new_layers = []
            for i,l in enumerate(self.incremental_layers):
                new_layers.append(copy.deepcopy(l))
            #make residual block
            new_block = self._residual_block(injection_index, new_layers, m,
                    self.member_number)
            new_model_config = self.model_config[:injection_index] + [new_block] + self.model_config[injection_index:]
            if self.freeze_old_layers:
                self.frozen_layers = range(0,injection_index)
            self.model_config = copy.deepcopy(new_model_config)
            self.weights = self.weights[:injection_index]
        self.member_number += 1
        return (m.to_yaml(), m.get_weights())

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = Resampler(dataset)
        self.weights = None
        self.member_number = 0
        self.frozen_layers = []
        with open(params.model_file, 'r') as model_file:
            model_yaml = model_file.read()
        self.model_config = keras.models.model_from_yaml(model_yaml).get_config()
        self.freeze_old_layers = False

    def serialize(self):
        self.set_defaults()
        return """
BARN {{
    n_epochs_after_first: {0},
    incremental_index: {1},
    incremental_layers: {2}
}}
        """.format(self.n_epochs_after_first,
                   self.incremental_index,
                   self.incremental_layers)


class AdaBoost_M1(EnsembleMethod):
    """
    Create an AdaBoost Ensemble from parameters
    """

    yaml_tag = u'!AdaBoostM1'

    def create_aggregator(self,params,members,train_set,valid_set):
            return WeightedAveragingRunner(members,self.alphas,params)

    def create_member(self):
        train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
        if self.member_number > 0 :
            resampled = [
                    train_set,
                    self.resampler.get_valid(),
                    self.resampler.get_test()
            ]
        else:
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        train_weights = None
        m = mlp.sequential_model(resampled, self.params,
                member_number = self.member_number)
        orig_train = self.resampler.get_train()
        errors = common.errors(m, orig_train[0], orig_train[1])
        e = np.sum((errors * self.D))
        alpha = .5 * math.log((1-e)/e)
        w = np.where(errors == 1,
            self.D * math.exp(alpha),
            self.D * math.exp(-alpha))
        self.D = w / w.sum()
        self.resampler.update_weights(self.D)
        self.alphas.append(alpha)
        self.member_number += 1
        return (m.to_yaml(), m.get_weights())

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = WeightedResampler(dataset)
        self.D = self.resampler.weights
        self.alphas = []
        self.member_number = 0

    def serialize(self):
        return 'AdaBoostM1'


class AdaBoost_M2(EnsembleMethod):
    """
    Create an AdaBoost Ensemble from parameters
    """

    yaml_tag = u'!AdaBoostM2'

    def create_aggregator(self,params,members,train_set,valid_set):
            return WeightedAveragingRunner(members,self.alphas,params)

    def create_member(self):
        train_weights = None
        if self.member_number > 0 :
            train_set, train_weights = self.resampler.make_new_train(self.params.resample_size)
            resampled = [
                    train_set,
                    self.resampler.get_valid(),
                    self.resampler.get_test()
            ]
        else:
            resampled = [
                self.resampler.get_train(),
                self.resampler.get_valid(),
                self.resampler.get_test()
            ]
        m = mlp.sequential_model(resampled, self.params,
                member_number = self.member_number,
                sample_weight = train_weights)
        orig_train = self.resampler.get_train()
        errors = common.errors(m, orig_train[0], orig_train[1])
        e = np.sum((errors * self.D))
        alpha = .5 * math.log((1-e)/e)
        w = np.where(errors == 1,
            self.D * math.exp(alpha),
            self.D * math.exp(-alpha))
        self.D = w / w.sum()
        self.resampler.update_weights(self.D)
        self.alphas.append(alpha)
        self.member_number += 1
        return (m.to_yaml(), m.get_weights())

    def prepare(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.resampler = WeightedResampler(dataset)
        self.D = self.resampler.weights
        self.alphas = []
        self.member_number = 0

    def serialize(self):
        return 'AdaBoostM2'


#class Stacking(EnsembleMethod):
#    """
#    Create a Stacking Runner from parameters
#    """
#
#    yaml_tag = u'!Stacking'
#
#    def __init__(self,n_hidden,update_rule,n_epochs,batch_size,learning_rate,
#            pretraining=None,pretraining_passes=1,training_method='normal',
#            L1_reg=0.0,L2_reg=0.0):
#        self.n_hidden = n_hidden
#        self.update_rule = update_rule
#        self.n_epochs = n_epochs
#        self.batch_size = batch_size
#        self.learning_rate = learning_rate
#        self.pretraining = pretraining
#        self.pretraining_passes = pretraining_passes
#        self.training_method = training_method
#        self.L1_reg = self.L1_reg
#        self.L2_reg = self.L2_reg
#
#    def create_aggregator(self,params,members,x,y,train_set,valid_set):
#        self.main_params = self.params
#        for p in self.params.__dict__:
#            if p not in self.__dict__:
#                self.__dict__[p] = self.params.__dict__[p]
#        return StackingRunner(members,x,y,train_set,valid_set,
#                Parameters(**self.__dict__))
#
#    def create_member(self,x,y):
#        resampled = [self.resampler.make_new_train(self.params.resample_size),
#                self.resampler.get_valid()]
#        pretraining_set = make_pretraining_set(resampled,self.params.pretraining)
#        self.params.member_number = len(self.members) + 1
#        m = mlp.test_mlp(resampled, self.params,
#                pretraining_set = pretraining_set, x=x, y=y)
#        w = m.get_weights()
#        self.members.append(w)
#        return m
#
#    def prepare(self, params, dataset):
#        self.params = params
#        self.dataset = dataset
#        self.resampler = Resampler(dataset)
#        self.members = []
#
#    def serialize(self):
#        return 'Stacking'
