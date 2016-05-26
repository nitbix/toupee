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
from theano.tensor.signal import pool
from theano.ifelse import ifelse
from utils import gaussian_filter
from data import sharedX,GPUTransformer
import weight_inits

floatX = theano.config.floatX

class Layer:
    def __init__(self,rng,inputs,n_in,n_out,activation,
                 dropout_rate,layer_name,W=None,b=None,weight_init=None):
        self.inputs = inputs
        self.dropout_rate=dropout_rate
        self.layer_name=layer_name
        self.activation = activation

        if weight_init is None:
            weight_init = weight_inits.GlorotWeightInit()
        if W is None:
            W = weight_init(rng,n_in,n_out,self.layer_name + '_W',self.activation)
        W = sharedX(W,name=self.layer_name + '_W')
        if b is None:
            b = weight_inits.ZeroWeightInit()(rng,n_out,None,layer_name + '_b',None)
        b = sharedX(b,name=self.layer_name + '_b')

        self.W = W
        self.b = b
        self.n_in = n_in
        self.n_out = n_out
        self.write_enable = 1.
        self.rejoin()

    def updates(self):
        return None

    def rejoin(self):
        self.y = T.dot(self.inputs, self.W) * (1. - self.dropout_rate) + self.b
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

    def set_input(self,inputs):
        self.inputs =inputs


class LCN(Layer):
    def __init__(self,inputs,kernel_size,x,y,channels,use_divisor=False):
        self.w = int(x)
        self.h = int(y)
        self.channels = channels
        self.use_divisor = use_divisor
        self.kernel_size = kernel_size
        self.n_in = self.channels * self.w * self.h
        self.n_out = self.channels * self.w * self.h
        self.W = sharedX(numpy.asarray([0.]))
        self.b = sharedX(numpy.asarray([0.]))
        self.write_enable = 0.
        self.dropout_rate = 0.
        self.layer_name = 'LCN'
        self.filter_kernel = gaussian_filter(self.kernel_size)
        self.inputs = inputs
        self.rejoin()

    def rejoin(self):
        self.instances = self.inputs.shape[0]
        self.out_shape = self.inputs.shape
        self.inputs = self.inputs.reshape([self.instances,self.channels,self.w,self.h],ndim=4)
        kernel_filter_shape = [1, self.channels, self.kernel_size, self.kernel_size]
        final_filter_shape = [self.channels, self.channels, self.kernel_size, self.kernel_size]
        filters,update = theano.scan(
                fn=lambda: self.filter_kernel,
                n_steps = self.channels
                )
        filters = filters.reshape(kernel_filter_shape,ndim=4)
        convout = T.nnet.conv2d(self.inputs, filters=filters, 
                filter_shape=final_filter_shape, border_mode='full')
        mid = int(numpy.floor(self.kernel_size/2.))
        new_X = self.inputs - convout[:,:,mid:-mid,mid:-mid]
        if self.use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = T.nnet.conv2d(T.sqr(T.abs_(new_X)), filters=filters, 
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)
            new_X /= divisor

        self.y = new_X.reshape(self.out_shape)
        self.params = []
        self.rebuild()

    def rebuild(self):
        self.output = self.y
        self.p_y_given_x = self.output

    def copy_weights(self,other):
        pass

    def set_weights(self,W,b):
        pass

class Elastic(Layer):
    def __init__(self,inputs,x,y,opts,channels,trainflag):
        self.inputs = inputs
        self.trainflag = trainflag
        self.w = int(x)
        self.h = int(y)
        self.channels = channels
        self.W = sharedX(numpy.asarray([0.]))
        self.b = sharedX(numpy.asarray([0.]))
        self.n_in = self.channels * self.w * self.h
        self.n_out = self.channels * self.w * self.h
        self.write_enable = 0.
        self.dropout_rate = 0.
        self.layer_name = 'elastic_transform'
        self.t = GPUTransformer(inputs,x,y,opts,channels,False,False)
        self.rejoin()

    def rejoin(self):
        self.y = T.switch(T.gt(self.trainflag,0.5),self.t.final_x,self.inputs)
        self.params = []
        self.rebuild()

    def rebuild(self):
        self.output = self.y
        self.p_y_given_x = self.output

    def copy_weights(self,other):
        pass

    def set_weights(self,W,b):
        pass

class Dropout(Layer):
    def __init__(self,rng,inputs,n_in,n_out,activation,
                 dropout_rate,layer_name,W=None,b=None,weight_init=None):
        self.inputs = inputs
        self.dropout_rate=dropout_rate
        self.layer_name=layer_name
        self.W = sharedX(numpy.asarray([0.]))
        self.b = sharedX(numpy.asarray([0.]))
        self.n_in = n_in
        self.n_out = n_in
        self.write_enable = 0.
        self.rejoin()

    def rejoin(self):
        self.y = self.inputs
        self.params = []
        self.rebuild()

    def rebuild(self):
        self.output = self.y
        self.p_y_given_x = self.output

    def copy_weights(self,other):
        pass

    def set_weights(self,W,b):
        pass

class FlatLayer(Layer):
    """
    Typical hidden layer of a MLP: units are fully-connected
    """

    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden',
                 weight_init=None):
        Layer.__init__(self,rng,inputs.flatten(ndim=2),n_in,n_out,activation,
                dropout_rate,layer_name,W,b,weight_init=weight_init)
        self.rebuild()

    def rebuild(self):
        self.output = (self.y if self.activation is None
                       else self.activation(self.y))
        self.p_y_given_x = self.output

    def rejoin(self):
        Layer.rejoin(self)
        self.rebuild()


class SoftMax(Layer):
    """
    SoftMax Layer
    """

    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden',
                 weight_init=None):
        Layer.__init__(self,rng,inputs.flatten(ndim=2),n_in,n_out,activation,
                dropout_rate,layer_name,W,b,weight_init=weight_init)
        self.rebuild()

    def rebuild(self):
        self.y = T.dot(self.inputs, self.W) * (1. - self.dropout_rate)
        self.params = [self.W]
        self.p_y_given_x = T.nnet.softmax(self.y)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

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
        self.y = T.dot(self.inputs, self.W) * (1. - self.dropout_rate)
        self.params = [self.W]
        self.rebuild()


class LogSoftMax(Layer):
    """
    SoftMax Layer
    """

    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh,dropout_rate=0,layer_name='hidden',
                 weight_init=None):
        Layer.__init__(self,rng,inputs.flatten(ndim=2),n_in,n_out,activation,
                dropout_rate,layer_name,W,b,weight_init=weight_init)
        self.rebuild()

    def rebuild(self):
        self.y = T.dot(self.inputs, self.W) * (1. - self.dropout_rate)
        self.params = [self.W]
        ydev = self.y - self.y.max(1,keepdims=True)
        self.p_y_given_x = ydev - T.log(T.sum(T.exp(ydev),axis=1,keepdims=True))
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

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
        self.y = T.dot(self.inputs, self.W) * (1. - self.dropout_rate)
        self.params = [self.W]
        self.rebuild()

class ConvFilter(Layer):
    """
    A Convolutional Layer, as per Convolutional Neural Networks. Includes
    filter, no pooling.
    """
    def __init__(self, rng, inputs, input_shape, filter_shape, W=None, b=None,
             activation=T.tanh,dropout_rate=0,layer_name='conv',
             border_mode='valid',weight_init=None):
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

        self.input_shape = input_shape #[batch_size,channels,y,x]
        self.filter_shape = filter_shape #[maps,channels,y,x]
        self.border_mode = border_mode
        self.fan_in = numpy.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * numpy.prod(self.filter_shape[2:])

        #W and b are slightly different for convnets - we need filter_shape
        if weight_init is None:
            weight_init = weight_inits.GlorotWeightInit()
        if W is None:
            W = weight_init(rng,self.fan_in,self.fan_out,layer_name + '_W',
                    activation, self.filter_shape)
        if b is None:
            b = weight_inits.ZeroWeightInit()(rng,self.filter_shape[0],None,
                    layer_name + '_b', None)
        Layer.__init__(self,rng,T.reshape(inputs,self.input_shape,ndim=4),self.filter_shape[0],
                self.filter_shape[1], activation,dropout_rate,layer_name,W,b)
        self.rebuild()

    def set_input(self,inputs):
        self.inputs = T.reshape(inputs,self.input_shape,ndim=4)

    def rebuild(self):
        self.delta_W = sharedX(
            value=numpy.zeros(self.filter_shape),
            name='{0}_delta_W'.format(self.layer_name))
        self.delta_b = sharedX(
            value=numpy.zeros_like(self.b.get_value(borrow=True)),
            name='{0}_delta_b'.format(self.layer_name))
        if self.border_mode == 'same':
            bm = 'full'
        else:
            bm = self.border_mode
        conv_out = T.nnet.conv2d(
            input=self.inputs,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.input_shape,
            border_mode=bm) * (1. - self.dropout_rate)

        if self.border_mode == 'same':
            shift_x = (self.filter_shape[2] - 1) // 2
            shift_y = (self.filter_shape[3] - 1) // 2
            self.conv_out = conv_out[:, :,
                                shift_x:self.input_shape[2] + shift_x,
                                shift_y:self.input_shape[3] + shift_y]
        else:
            self.conv_out = conv_out

        self.y_out = self.activation(self.conv_out + self.b.dimshuffle('x',0,'x','x'))
        self.output = self.y_out
        self.params = [self.W, self.b]

    def rejoin(self):
        self.rebuild()

class ConvolutionalLayer(Layer):
    """
    A Convolutional Layer, as per Convolutional Neural Networks. Includes filter, and pooling.
    """
    def __init__(self, rng, inputs, input_shape, filter_shape, pool_size, W=None, b=None,
             activation=T.tanh,dropout_rate=0,layer_name='conv',
             border_mode='valid',pooling='max',weight_init=None):
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

        self.input_shape = input_shape #[batch_size,channels,y,x]
        self.filter_shape = filter_shape #[maps,channels,y,x]
        self.pooling = pooling
        self.pool_size = pool_size #[y,x]
        self.border_mode = border_mode
        self.fan_in = numpy.prod(self.filter_shape[1:])
        self.fan_out = self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(pool_size)

        #W and b are slightly different for convnets - we need filter_shape
        if weight_init is None:
            weight_init = weight_inits.GlorotWeightInit()
        if W is None:
            W = weight_init(rng,self.fan_in,self.fan_out,layer_name + '_W',
                    activation, self.filter_shape)
        if b is None:
            b = weight_inits.ZeroWeightInit()(rng,self.filter_shape[0],None,
                    layer_name + '_b', None)
        Layer.__init__(self,rng,T.reshape(inputs,self.input_shape,ndim=4),self.filter_shape[0],
                self.filter_shape[1], activation,dropout_rate,layer_name,W,b)
        self.rebuild()

    def set_input(self,inputs):
        self.inputs = T.reshape(inputs,self.input_shape,ndim=4)

    def rebuild(self):
        self.delta_W = sharedX(
            value=numpy.zeros(self.filter_shape),
            name='{0}_delta_W'.format(self.layer_name))
        self.delta_b = sharedX(
            value=numpy.zeros_like(self.b.get_value(borrow=True)),
            name='{0}_delta_b'.format(self.layer_name))
        if self.border_mode == 'same':
            bm = 'full'
        else:
            bm = self.border_mode
        conv_out = T.nnet.conv2d(
            input=self.inputs,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.input_shape,
            border_mode=bm) * (1. - self.dropout_rate)

        if self.border_mode == 'same':
            shift_x = (self.filter_shape[2] - 1) // 2
            shift_y = (self.filter_shape[3] - 1) // 2
            self.conv_out = conv_out[:, :,
                                shift_x:self.input_shape[2] + shift_x,
                                shift_y:self.input_shape[3] + shift_y]
        else:
            self.conv_out = conv_out

        self.y_out = self.activation(self.conv_out + self.b.dimshuffle('x',0,'x','x'))
        self.pooled_out = pool.pool_2d(input=self.y_out, 
                                                 ds=self.pool_size,
                                                 ignore_border=True,
                                                 mode=self.pooling)
        self.output = self.pooled_out
        self.params = [self.W, self.b]

    def rejoin(self):
        self.rebuild()
