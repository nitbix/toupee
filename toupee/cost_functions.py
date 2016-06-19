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
import theano.tensor.extra_ops as TE

class CostFunction(yaml.YAMLObject):

    def __call__(self,model,y):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

class CrossEntropy(CostFunction):
    yaml_tag = u'!CrossEntropy'
    def __call__(self,model,y):
        """
        Return the categorical cross-entropy of the prediction
        of this model under a given target distribution.
        """
        if y.dtype.startswith('int'):
            return T.nnet.categorical_crossentropy(model.p_y_given_x,y).mean()
        else:
            return T.nnet.crossentropy(model.p_y_given_x,y).mean()

    def serialize(self):
        return 'CrossEntropy'

class NegLogLikelihood(CostFunction):
    yaml_tag = u'!NegLogLikelihood'
    def __call__(self,model,y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(model.p_y_given_x)[T.arange(y.shape[0]), y])

    def serialize(self):
        return 'NegLogLikelihood'

class MSE(CostFunction):
    yaml_tag = u'!MSE'
    def __call__(self,model,y):
        """
        Return the mean squared error
        """

        if y.ndim != model.y.ndim:
            raise TypeError('y should have the same shape as model.y',
                ('y', y.type, 'y_pred', model.y.type, 'layer', model.layer_name))
        return T.mean((model.p_y_given_x - y) ** 2)

    def serialize(self):
        return 'MSE'

class CategoricalMSE(CostFunction):
    yaml_tag = u'!CategoricalMSE'
    def __call__(self,model,y):
        """
        Return the mean squared error
        """

        one_hot = TE.to_one_hot(y,model.n_out)
        if one_hot.ndim != model.y.ndim:
            raise TypeError('y one_hot representation should have the same shape as model.y',
                ('y', y.type, 'y_pred', model.y.type, 'layer', model.layer_name))
        return T.mean((model.p_y_given_x - one_hot) ** 2)

    def serialize(self):
        return 'CategoricalMSE'
