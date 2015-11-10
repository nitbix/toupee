#!/usr/bin/python
from __future__ import print_function

"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import gc
import sys
import numpy as np
import scipy.ndimage as ni
import numpy.random
import theano
import theano.tensor as T
import gzip
import cPickle
import math
from skimage import transform as tf
import multiprocessing
import theano.tensor.signal.conv as sigconv
from scipy.misc import imsave

from theano.sandbox.rng_mrg import MRG_RandomStreams

floatX = theano.config.floatX
rng = numpy.random.RandomState(42)
theano_rng = MRG_RandomStreams(max(rng.randint(2 ** 15), 1))

#import matplotlib.pyplot as plt

def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a shared variable of type floatX
    borrowed from pylearn2
    """

    if dtype is None:
        dtype = floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def sharedXscalar(value, name=None, borrow=False):
    """
    Transform value into a shared variable scalar
    """

    return theano.shared(floatX(value), name=name, borrow=borrow)

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def mask(p,shape,dtype=floatX):
    return theano_rng.binomial(p=p, size=shape, dtype=dtype)

def corrupt(data,p):
    return mask(1-p,data.shape,dtype=floatX) * data

def load_data(dataset, resize_to=None, shared=True, pickled=True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    data_dir, data_file = os.path.split(dataset)
    if pickled:
        if data_dir == "" and not os.path.isfile(dataset):
            new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path
        print('... loading data')
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    else:
        tr = np.load(dataset + 'train.npz')
        v = np.load(dataset + 'valid.npz')
        te = np.load(dataset + 'test.npz')
        train_set = (tr['x'],tr['y'])
        valid_set = (v['x'],v['y'])
        test_set = (te['x'],te['y'])
    if resize_to is not None:
        orig_size = math.sqrt(train_set[0].shape[1])
        train_set = (
                        pad_dataset(
                            train_set[0].reshape((train_set[0].shape[0],
                                                    orig_size,orig_size)
                            ),
                            resize_to),
                        train_set[1])
        valid_set = (
                        pad_dataset(
                            valid_set[0].reshape((valid_set[0].shape[0],
                                                    orig_size,orig_size)
                            ),
                            resize_to),
                        valid_set[1])
        test_set  = (
                        pad_dataset(
                            test_set[0].reshape((test_set[0].shape[0],
                                                    orig_size,orig_size)
                            ),
                            resize_to),
                        test_set[1])
    if shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def make_pretraining_set(datasets,mode):
    return (datasets[0][0],datasets[0][1])


class Resampler:
    """
    Resample a dataset either uniformly or with a given probability
    distribution
    """

    def __init__(self,dataset):
        self.train,self.valid,self.test = dataset
        self.train_x, self.train_y = self.train
        self.valid_x, self.valid_y = self.valid
        self.test_x, self.test_y = self.test
        self.train_size = len(self.train_x)
        self.s_train = None
        self.s_valid = None
        self.s_test = None
        
    def make_new_train(self,sample_size,distribution=None):
        if distribution is None:
            sample = numpy.random.randint(low=0,high=self.train_size,size=sample_size)
        else:
            raise Exception("not implemented");
        sampled_x = []
        sampled_y = []
        for s in sample:
            sampled_x.append(self.train_x[s])
            sampled_y.append(self.train_y[s])
        return shared_dataset((sampled_x,sampled_y))

    def get_train(self):
        if self.s_train is None:
            self.s_train = shared_dataset(self.train)
        return self.s_train

    def get_valid(self):
        if self.s_valid is None:
            self.s_valid = shared_dataset(self.valid)
        return self.s_valid

    def get_test(self):
        if self.s_test is None:
            self.s_test = shared_dataset(self.test)
        return self.s_test


class WeightedResampler(Resampler):
#WORK IN PROGRESS

    def update_weights(self,new_weights):
        pass

    def make_new_train(self,sample_size):
        pass

def transform_aux_map(tr,x):
    return tr.apply(x)

def f(x):
    return x

def pad_dataset(xval,end_size):
    """ 
    Thanks to https://github.com/ilyakava/ciresan
    """
    new_x = []
    for x in xval:
        cs = x.shape[0]
        padding = end_size - cs
        bp = round(padding / 2) # before padding (left)
        ap = padding - bp # after padding (right)
        pads = (bp,ap)
        if bp + ap > 0:
            new_x.append(numpy.pad(x,(pads,pads),mode='constant').reshape(end_size**2))
        else: # image is too big now, unpad/slice
            si = -bp # start index
            ei = cs + ap # end index
            new_x.append(x[si:ei, si:ei].reshape(end_size**2))
    return numpy.asarray(new_x)

class Transformer:
    """
    Apply translation, scaling, rotation and other transformations to a 
    training set to produce a larger, noisy training set
    """

    def __init__(self,original_set,x,y,alpha,beta,gamma,sigma,noise_var,progress = False):
        print("..transforming dataset")
        self.progress = progress
        self.x = x
        self.y = y
        self.min_trans_x = -2
        self.max_trans_x =  2
        self.min_trans_y = -2
        self.max_trans_y =  2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.noise_var = noise_var
        self.original_x = numpy.asarray(original_set)
        self.final_x = []
        self.instance_no = 0
        instances = len(self.original_x)
        rng = numpy.random.RandomState(42)
        self.original_x = numpy.asarray(self.original_x).reshape(instances,self.x,self.y)
        p = multiprocessing.Pool(32)
        deferred = [p.apply_async(transform_aux_map,args=(self,x)) for x in
                self.original_x]
        self.final_x = [x.get() for x in deferred]

    def apply(self,curr_x):
#        if self.progress and self.instance_no % 100 == 0:
#            print("instance {0}".format(self.instance_no), end="\r")
        dx = numpy.random.uniform(low=self.min_trans_x,high=self.max_trans_x)
        dy = numpy.random.uniform(low=self.min_trans_y,high=self.max_trans_y)
        curr_x = self.translate_instance(curr_x,int(dx),int(dy))
        angle = numpy.random.uniform(low=-self.beta,high=self.beta)
        shear = numpy.random.uniform(low=-self.beta,high=self.beta)
        curr_x = self.rotate_instance(curr_x,angle)
        #curr_x = self.gaussian_noise(curr_x,noise_var)
        scale_x = 1. + numpy.random.uniform(low=-self.gamma,high=self.gamma) / 100.
        scale_y = 1. + numpy.random.uniform(low=-self.gamma,high=self.gamma) / 100.
        curr_x = self.scale(curr_x,[scale_x,scale_y])
#            curr_x = self.elastic_transform(curr_x,sigma,alpha)
#        trans = tf.AffineTransform(
#                    scale=(scale_x,scale_y),
#                    shear=shear,
#                    rotation=angle,
#                    translation=(dx,dy)
#                )
#        warped_x = tf.warp(curr_x,trans)
#        if plot:
#            plt.gray()
#            plt.imshow(warped_x)
#            plt.show()
#        self.instance_no += 1
        warped_x = curr_x
        return warped_x.flatten()

    def elastic_transform(self,xval,sigma,alpha):
            field_x = numpy.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
            field_y = numpy.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
            convolved_field_x = ni.filters.gaussian_filter(field_x,sigma)
            convolved_field_y = ni.filters.gaussian_filter(field_y,sigma)
            convolved_field_x = convolved_field_x * alpha / max(abs(convolved_field_x.flatten()))
            convolved_field_y = convolved_field_y * alpha / max(abs(convolved_field_y.flatten()))
            def mapping(coords):
                x,y = coords
                return  (x+convolved_field_x[x,y],y+convolved_field_y[x,y])
            return ni.geometric_transform(xval,mapping)

    def translate_instance(self,xval, dx, dy):
        return np.roll(np.roll(xval,dx,axis=0),dy,axis=1)

    def rotate_instance(self,xval,angle):
        return tf.rotate(xval,angle, mode='reflect')

    def gaussian_noise(self,xval,sigma):
        return xval + numpy.random.normal(0.,sigma,xval.shape)

    def scale(self,xval,scaling):
        return ni.zoom(xval,scaling)

    def get_data(self):
        return np.array(self.final_x)

class GPUTransformer:
    """
    Apply translation, scaling, rotation and other transformations to a 
    training set to produce a larger, noisy training set, using Theano on the
    GPU
    Credit for this function to theanet https://github.com/rakeshvar/theanet/
    """

    def __init__(self,original_set,x,y,opts,
                    layers=1,progress=False,save=False):
        print("..transforming dataset")
        self.min_trans_x = -2
        self.max_trans_x =  2
        self.min_trans_y = -2
        self.max_trans_y =  2
        self.alpha = opts['alpha']
        self.beta  = opts['beta']
        self.gamma = opts['gamma']
        self.sigma = opts['sigma']
        self.pflip = opts['pflip']
        self.translation = opts['translation']
        self.bilinear = opts['bilinear']
        self.invert = opts['invert'] if 'invert' in opts else False
        self.center_uncertainty = opts['center_uncertainty'] if 'center_uncertainty' in opts else 0.
        self.x = x
        self.y = y
        self.layers = layers
        self.original_x = original_set
        self.instances = self.original_x.shape[0].eval()
        inpt = self.original_x.reshape([self.instances,self.layers,self.x,self.y])

        srs = T.shared_randomstreams.RandomStreams(rng.randint(1e6))
        target = T.as_tensor_variable(np.indices((self.y, self.x)).astype('float32'))

        # Translate
        transln = self.translation * srs.uniform((2, 1, 1), -1,dtype=floatX)
        target += transln

        # Build a gaussian filter
        var = self.sigma ** 2
        filt = np.array([[np.exp(-.5 * (i * i + j * j) / var)
                         for i in range(-self.sigma, self.sigma + 1)]
                         for j in range(-self.sigma, self.sigma + 1)], dtype=floatX)
        filt /= 2 * np.pi * var

        # Elastic
        elast = self.alpha * srs.normal((2, self.y, self.x),dtype=floatX)
        elast = sigconv.conv2d(elast, filt, (2, self.y, self.x), filt.shape, 'full')
        elast = elast[:, self.sigma:self.y + self.sigma, self.sigma:self.x + self.sigma]
        target += elast

        # Center at 'about' half way
        origin = srs.uniform((2, 1, 1), 0.5 - self.center_uncertainty,
                 0.5 + self.center_uncertainty,dtype=floatX) * \
                 np.array((self.y, self.x)).reshape((2, 1, 1)).astype('float32')
        target -= origin

        # Zoom
        zoomer = T.exp(np.log(1. + (self.gamma/100.)).astype('float32') * srs.uniform((2, 1, 1), -1,dtype=floatX))
        target *= zoomer

        # Rotate
        theta = (self.beta * np.pi / 180) * srs.uniform(low=-1,dtype=floatX)
        c, s = T.cos(theta), T.sin(theta)
        rotate = T.stack(c, -s, s, c).reshape((2,2))
        target = T.tensordot(rotate, target, axes=((0, 0)))

        # Uncenter
        target += origin

        # Clip the mapping to valid range and linearly interpolate
        transy = T.clip(target[0], 0, self.y - 1 - .001)
        transx = T.clip(target[1], 0, self.x - 1 - .001)

        if self.bilinear:
            topp = T.cast(transy, 'int32')
            left = T.cast(transx, 'int32')
            fraction_y = T.cast(transy - T.cast(topp, floatX), floatX)
            fraction_x = T.cast(transx - T.cast(left, floatX), floatX)

            output = inpt[:, :, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                     inpt[:, :, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                     inpt[:, :, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                     inpt[:, :, topp + 1, left + 1] * fraction_y * fraction_x
        else:
            vert = T.iround(transy)
            horz = T.iround(transx)
            output = inpt[:, :, vert, horz]

        # Now add some noise
        mask = srs.binomial(n=1, p=self.pflip, size=inpt.shape, dtype=floatX)
        acc_x = (1 - output) * mask + output * (1 - mask)

        if self.invert:
            acc_x = 1. - acc_x

        self.final_x = acc_x

        if save:
            self.save_images()

    def save_images(self):
        to_save = self.final_x.reshape([self.instances,self.x,self.y]).eval({})
        for i,x in enumerate(to_save[:100]):
            print(i)
            imsave('trans{0}.png'.format(i),x)
        to_save = self.original_x.reshape([self.instances,self.x,self.y]).eval({})
        for i,x in enumerate(to_save[:100]):
            print(i)
            imsave('orig{0}.png'.format(i),x)

    def get_data(self):
        return sharedX(self.final_x.reshape([self.instances,self.layers * self.x * self.y]).eval())

def one_hot(dataset):
    b = np.zeros((dataset.size, dataset.max()+1),dtype=floatX)
    b[np.arange(dataset.size), dataset] = 1.
    return b
