#!/usr/bin/python
from __future__ import print_function

"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
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
#import matplotlib.pyplot as plt

def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a shared variable of type floatX
    borrowed from pylearn2
    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data(dataset, shared=True, pickled=True):
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
    pretraining_set = None
    if mode == 'unsupervised':
        pretraining_set = datasets[0][0]
    elif mode == 'supervised':
        pretraining_set = datasets[0]
    elif mode == 'both':
        pretraining_set = (datasets[0][0],datasets[0][1],datasets[0][0])
    return pretraining_set


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



class Transformer:
    """
    Apply translation, scaling, rotation and other transformations to a 
    training set to produce a larger, noisy training set
    """

    def __init__(self,original_set,x,y, progress = False):
        print("..transforming dataset")
        self.progress = progress
        self.x = x
        self.y = y
        min_trans_x = -2
        max_trans_x =  2
        x_step = 2
        min_trans_y = -2
        max_trans_y =  2
        y_step = 2
        min_angle = -20
        max_angle =  20
        angle_step = 20
        sigmas = [0.2,0.4]
        gaussian_resamples = 1
        scalings = [0.8,1.2]
        self.original_x, self.original_y = original_set
        self.final_x = []
        self.final_y = []
        self.instance_no = 0
        instances = len(self.original_x)
        elastic_sigma = 6.
        elastic_alpha = 5.
        elastic_transforms = 2
        rng = numpy.random.RandomState(42)
        for i in xrange(0,instances):
            self.step_no = 0
            curr_x = self.original_x[i].reshape(self.x,self.y)
            curr_y = self.original_y[i]
            for dx in xrange(min_trans_x,max_trans_x,x_step):
                for dy in xrange(min_trans_y,max_trans_y,y_step):
                    if dx != 0 or dy != 0:
                        self.add_instance(
                                self.translate_instance(curr_x,dx,dy),
                                curr_y)
            for angle in xrange(min_angle,max_angle,angle_step):
                if angle != 0:
                    self.add_instance(self.rotate_instance(curr_x,angle),curr_y)
            for j in xrange(1,gaussian_resamples):
                for sigma in sigmas:
                    self.add_instance(self.gaussian_noise(curr_x,sigma),curr_y)
#            for scale_x in scalings:
#                for scale_y in scalings:
#                    if scale_x != 1 or scale_y != 1:
#                        self.add_instance(self.scale(curr_x,[scale_x,scale_y]),curr_y)
            for j in range(0,elastic_transforms):
                self.add_instance(self.elastic_transform(curr_x,elastic_sigma,elastic_alpha),curr_y)

            self.instance_no += 1
            gc.collect()

    def elastic_transform(self,xval,sigma,alpha):
            field_x = numpy.random.rand(28,28) * 2. - 1.
            field_y = numpy.random.rand(28,28) * 2. - 1.
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
        return ni.rotate(xval,angle,reshape=False)

    def gaussian_noise(self,xval,sigma):
        return xval + numpy.random.normal(0.,sigma,xval.shape)

    def scale(self,xval,scaling):
        return ni.zoom(xval,scaling)

    def add_instance(self,xval,yval,plot=False):
#        if plot:
#            plt.gray()
#            plt.imshow(xval)
#            plt.show()
        self.final_x.append(xval.flatten())
        self.final_y.append(yval)
        self.step_no += 1
        if self.progress:
            print("instance {0}, step {1}".format(
                    self.instance_no, self.step_no), end="\r")

    def get_data(self):
        return (np.array(self.final_x),np.array(self.final_y))

    def transform_dataset(dataset):
        train,valid,test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        aggregate_x = np.concatenate((train_x, valid_x), axis=0)
        aggregate_y = np.concatenate((train_y, valid_y), axis=0)
        t = Transformer((aggregate_x,aggregate_y),28,28)
        aggregate_train = t.get_data()
        aggregate_valid = (aggregate_x, aggregate_y)
        return (aggregate_train,aggregate_valid,test)
