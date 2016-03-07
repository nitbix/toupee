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
import theano.printing
import yaml
from data import sharedX
import common

class LearningRate(yaml.YAMLObject):

    def get(self):
        raise NotImplementedError()

    def reset(self,updates):
        raise NotImplementedError()

    def epoch_hook(self,updates):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

class CompositeLearningRate(LearningRate):
    yaml_tag = u'!CompositeLearningRate'

    def __new__(cls):
        instance = super(CompositeLearningRate,cls).__new__(cls)
        common.toupee_global_instance.add_epoch_hook(lambda x: instance.epoch_hook(x))
        common.toupee_global_instance.add_reset_hook(lambda x: instance.reset(x))
        return instance

    def serialize(self):
        schedule_str = ["{0} - {1}".format(e,x.serialize())
                        for e,x in self.schedule]
        return 'CompositeLearningRate: {{ {0} }}'.format(",".join(schedule_str))

    def reset(self,updates):
        if 'current_epoch' not in self.__dict__:
            self.current_epoch = 1
        self.active_lr = self.schedule[1]
        updates = []
        for e,lr in self.schedule.iteritems():
            lr.reset(updates)

    def epoch_hook(self,updates):
        epoch = self.current_epoch + 1
        if epoch in self.schedule.keys():
            self.active_lr = self.schedule[epoch]
        self.active_lr.epoch_hook(updates)

    def get(self):
        return self.active_lr.get()

class FixedLearningRate(LearningRate):

    yaml_tag = u'!FixedLearningRate'

    def get(self):
        if 'shared_rate' not in self.__dict__:
            self.shared_rate = sharedX(self.rate)
        return self.shared_rate

    def reset(self,updates):
        pass

    def epoch_hook(self,updates):
        pass

    def serialize(self):
        return "FixedLearningRate { rate: {0} }".format(self.rate)

class LinearDecayLearningRate(LearningRate):

    yaml_tag = u'!LinearDecayLearningRate'

    def __new__(cls):
        instance = super(LinearDecayLearningRate,cls).__new__(cls)
        common.toupee_global_instance.add_epoch_hook(lambda x: instance.epoch_hook(x))
        common.toupee_global_instance.add_reset_hook(lambda x: instance.reset(x))
        return instance

    def get(self):
        if 'current_rate' not in self.__dict__:
            raise Exception("Uninitialised LinearDecayLearningRate")
        return self.current_rate

    def reset(self,updates):
        if 'current_epoch' not in self.__dict__:
            self.current_epoch = sharedX(1.)
        if 'current_rate' not in self.__dict__:
            self.current_rate = sharedX(self.start,borrow=True)
        updates.append((self.current_rate,self.start))
        updates.append((self.current_epoch,1.))

    def epoch_hook(self,updates):
        if 'current_epoch' not in self.__dict__:
            self.current_epoch = sharedX(1.)
        epoch = self.current_epoch + 1
        new_rate = (self.start + ((self.stop - self.start) / self.steps) *
                    T.clip(epoch,1,self.steps))
        updates.append((self.current_rate,new_rate))
        updates.append((self.current_epoch,epoch))

    def serialize(self):
        return "LinearDecayLearningRate { start: {0} , stop: {1} , steps: {2}}".format(
                self.start, self.stop, self.steps)

class MultiplicativeDecayLearningRate(LearningRate):

    yaml_tag = u'!MultiplicativeDecayLearningRate'

    def __new__(cls):
        instance = super(MultiplicativeDecayLearningRate,cls).__new__(cls)
        common.toupee_global_instance.add_epoch_hook(lambda x: instance.epoch_hook(x))
        common.toupee_global_instance.add_reset_hook(lambda x: instance.reset(x))
        return instance

    def get(self):
        if 'current_rate' not in self.__dict__:
            raise Exception("Uninitialised MultiplicativeDecayLearningRate")
        return self.current_rate

    def reset(self,updates):
        if 'current_rate' not in self.__dict__:
            self.current_rate = sharedX(self.start,borrow=True)
        updates.append((self.current_rate,self.start))

    def epoch_hook(self,updates):
        if 'multiplier' not in self.__dict__:
            self.multiplier = 0.9
        new_rate = T.clip(self.current_rate * self.multiplier,
                            self.stop,
                            self.start)
        updates.append((self.current_rate,new_rate))

    def serialize(self):
        return "MultiplicativeDecayLearningRate {{ start: {0} , stop: {1} , multiplier: {2} }}".format(
                self.start, self.stop, self.multiplier)
