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
import yaml
from data import sharedX

class UpdateRule(yaml.YAMLObject):

    def __call__(self, param, learning_rate, gparam, mask, updates,
                 current_cost, previous_cost):
        raise NotImplementedError()

class SGD(UpdateRule):

    yaml_tag = u'!SGD'
    def __init__(self):
        pass

    def __call__(self, param, learning_rate, gparam, mask, updates,
                 current_cost, previous_cost):
        if 'momentum' not in self.__dict__:
            self.momentum = sharedX(0.)
        if 'linear_decay' not in self.__dict__:
            self.linear_decay = {'stop': sharedX(0.),
                                 'start': sharedX(learning_rate),
                                 'steps': sharedX(1.)}
        self.velocity = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
        self.current_step = sharedX(1,'current_step')
        epsilon = self.linear_decay['start'] + (
                   (self.linear_decay['start'] - self.linear_decay['stop']) /
                    self.linear_decay['steps']
                  ) * T.clip(self.current_step,1,self.linear_decay['steps'])
        velocity = (self.velocity * self.momentum -
                                  epsilon * gparam * mask)
        updates.append((self.velocity,velocity))
        updates.append((self.current_step,self.current_step + 1))
        return param - velocity

class RPropVariant(UpdateRule):

    def __init__(self):
        self.eta_plus = 1.2
        self.eta_minus = 0.5
        self.max_delta=50
        self.min_delta=1e-6

    def __init__(self,eta_plus,eta_minus,max_delta,min_delta):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.max_delta = max_delta
        self.min_delta = min_delta

    def __repr__(self):
        return "%s(eta_plus=%r,eta_minus=%r,max_delta=%r,min_delta=%r)" % (
                self.__class__.__name__, self.eta_plus,self.eta_minus,
                self.max_delta,self.min_delta)
                
class OldRProp(RPropVariant):

    yaml_tag = u'!OldRProp'
    def __call__(self, param, learning_rate, gparam, mask, updates,
                 current_cost, previous_cost):
        previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
        delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
        previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
        zero = T.zeros_like(param)
        one = T.ones_like(param)
        change = previous_grad * gparam

        new_delta = T.clip(
                T.switch(
                    T.gt(change,0.),
                    delta * self.eta_plus,
                    T.switch(
                        T.lt(change,0.),
                        delta * self.eta_minus,
                        delta
                    )
                ),
                self.min_delta,
                self.max_delta
        )
        new_previous_grad = T.switch(
                T.gt(change,0.),
                gparam,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    gparam
                )
        )
        inc = T.switch(
                T.gt(change,0.),
                - T.sgn(gparam) * new_delta,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    - T.sgn(gparam) * new_delta
                )
        )

        updates.append((previous_grad,new_previous_grad))
        updates.append((delta,new_delta))
        updates.append((previous_inc,inc))
        return param + inc * mask


class RProp(RPropVariant):

    yaml_tag = u'!RProp'
    def __init__(self):
        self.eta_plus = 1.01
        self.eta_minus = 0.1
        self.max_delta=5
        self.min_delta=1e-3

    def __call__(self, param, learning_rate, gparam, mask, updates,
                 current_cost, previous_cost):
        previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
        delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
        previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
        zero = T.zeros_like(param)
        one = T.ones_like(param)
        change = previous_grad * gparam

        new_delta = T.clip(
                T.switch(
                    T.eq(gparam,0.),
                    delta,
                    T.switch(
                        T.gt(change,0.),
                        delta * self.eta_plus,
                        T.switch(
                            T.lt(change,0.),
                            delta * self.eta_minus,
                            delta
                        )
                    )
                ),
                self.min_delta,
                self.max_delta
        )
        new_previous_grad = T.switch(
                T.eq(mask * gparam,0.),
                previous_grad,
                T.switch(
                    T.gt(change,0.),
                    gparam,
                    T.switch(
                        T.lt(change,0.),
                        zero,
                        gparam
                    )
                )
        )
        inc = T.switch(
                T.eq(mask * gparam,0.),
                zero,
                T.switch(
                    T.gt(change,0.),
                    - T.sgn(gparam) * new_delta,
                    T.switch(
                        T.lt(change,0.),
                        zero,
                        - T.sgn(gparam) * new_delta
                    )
                )
        )

        updates.append((previous_grad,new_previous_grad))
        updates.append((delta,new_delta))
        updates.append((previous_inc,inc))
        return param + inc * mask

class iRProp(RPropVariant):

    yaml_tag = u'!iRProp'
    def __init__(self):
        self.eta_plus = 1.5
        self.eta_minus = 0.25
        self.max_delta=500
        self.min_delta=1e-8

    def __call__(self, param, learning_rate, gparam, mask, updates,
                 current_cost, previous_cost):
        previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
        delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
        previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
        zero = T.zeros_like(param)
        one = T.ones_like(param)
        change = previous_grad * gparam

        new_delta = T.clip(
                T.switch(
                    T.eq(mask * gparam,0.),
                    delta,
                    T.switch(
                        T.gt(change,0.),
                        delta * self.eta_plus,
                        T.switch(
                            T.lt(change,0.),
                            delta * self.eta_minus,
                            delta
                        )
                    )
                ),
                self.min_delta,
                self.max_delta
        )
        new_previous_grad = T.switch(
                T.eq(mask * gparam,0.),
                previous_grad,
                T.switch(
                    T.gt(change,0.),
                    gparam,
                    T.switch(
                        T.lt(change,0.),
                        zero,
                        gparam
                    )
                )
        )
        inc = T.switch(
                T.eq(mask * gparam,0.),
                zero,
                T.switch(
                    T.gt(change,0.),
                    - T.sgn(gparam) * new_delta,
                    T.switch(
                        T.lt(change,0.),
                        zero,
#                    - T.sgn(gparam) * new_delta
                        T.switch( 
                            T.gt(current_cost, previous_cost),
                            - T.sgn(gparam) * new_delta,
                            zero
                        )
                    )
                )
        )

        updates.append((previous_grad,new_previous_grad))
        updates.append((delta,new_delta))
        updates.append((previous_inc,inc))
        return param + inc * mask
