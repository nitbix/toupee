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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from data import sharedX

floatX = theano.config.floatX


class Layer:
    def __init__(self,rng,inputs,n_in,n_out,activation,
                 dropout_rate,layer_name,W=None,b=None):
        self.inputs = inputs
        self.dropout_rate=dropout_rate
        self.layer_name=layer_name
        self.activation = activation

        #TODO: initialisation functions need to be separate
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name=layer_name + '_W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=layer_name + '_b', borrow=True)

        self.W = W
        self.b = b
        self.n_in = n_in
        self.n_out = n_out
        self.y = T.dot(self.inputs, self.W) * (1. / (1. - self.dropout_rate)) + self.b
        self.params = [self.W, self.b]

    def rejoin(self):
        self.y = T.dot(self.inputs, self.W) + self.b
        self.params = [self.W, self.b]

    def rebuild(self):
        raise NotImplementedError()

    def copy_weights(self,other):
        self.W = sharedX(other.W.get_value())
        self.b = sharedX(other.b.get_value())
        self.rebuild()

    def set_weights(self,W,b):
        self.W = sharedX(W)
        self.b = sharedX(b)
        self.rebuild()


class FlatLayer(Layer):
    """
    Typical hidden layer of a MLP: units are fully-connected
    """

    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden'):

        Layer.__init__(self,rng,inputs.flatten(ndim=2),n_in,n_out,activation,dropout_rate,layer_name,W,b)
        self.rebuild()

    def rebuild(self):
        self.output = (self.y if self.activation is None
                       else self.activation(self.y))

    def rejoin(self):
        Layer.rejoin(self)
        self.rebuild()


class SoftMax(Layer):
    """
    SoftMax Layer
    """

    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden'):
        W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=floatX),
                               name='W', borrow=True)
        b = theano.shared(value=numpy.zeros((n_out,), dtype=floatX),
                               name='b', borrow=True)
        Layer.__init__(self,rng,inputs.flatten(ndim=2),n_in,n_out,activation,dropout_rate,layer_name,W,b)
        self.rebuild()

    def rebuild(self):
        self.p_y_given_x = T.nnet.softmax(self.y)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y = self.p_y_given_x

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y), dtype=floatX, acc_dtype=floatX)
        else:
            raise NotImplementedError()

    def rejoin(self):
        Layer.rejoin(self)
        self.rebuild()


class ConvolutionalLayer(Layer):
    """
    A Convolutional Layer, as per Convolutional Neural Networks. Includes filter, and pooling.
    """
    #TODO: rejoin and rebuild
    def __init__(self, rng, inputs, input_shape, filter_shape, pool_size, W=None, b=None,
             activation=T.tanh,dropout_rate=0,layer_name='conv',border_mode='valid',pooling='max'):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        assert input_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pooling = pooling
        self.pool_size = pool_size
        self.border_mode = border_mode
        self.fan_in = numpy.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(pool_size)

        #W and b are slightly different
        if W is None:
                W_bound = numpy.sqrt(6. / (self.fan_in + self.fan_out))
                initial_W = numpy.asarray( rng.uniform(
                                       low=-W_bound, high=W_bound,
                                       size=self.filter_shape),
                                       dtype=theano.config.floatX)

                if activation == T.nnet.sigmoid:
                    initial_W *= 4
                W = theano.shared(value = initial_W, name = 'W')
        if b is None:
                b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b')

        Layer.__init__(self,rng,T.reshape(inputs,input_shape,ndim=4),self.filter_shape[0],
                self.filter_shape[1], activation,dropout_rate,layer_name,W,b)
        self.rebuild()

    def rebuild(self):
        self.delta_W = sharedX(
            value=numpy.zeros(self.filter_shape),
            name='{0}_delta_W'.format(self.layer_name))
        self.delta_b = sharedX(
            value=numpy.zeros_like(self.b.get_value(borrow=True)),
            name='{0}_delta_b'.format(self.layer_name))
        self.conv_out = conv.conv2d(
            input=self.inputs,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.input_shape,
            border_mode=self.border_mode) * (1. / (1. - self.dropout_rate))
        self.y_out = self.activation(self.conv_out + self.b.dimshuffle('x',0,'x','x'))
        self.pooled_out = downsample.max_pool_2d(input=self.y_out, 
                                                 ds=self.pool_size,
                                                 ignore_border=True,
                                                 mode=self.pooling)
        self.output = self.pooled_out

    def rejoin(self):
        self.rebuild()
