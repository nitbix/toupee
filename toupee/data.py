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
import scipy.stats
import numpy.random
import gzip
import pickle
import math
from skimage import transform as tf
import multiprocessing

def corrupt(data,p):
    return mask(1-p,data.shape,dtype=floatX) * data

def sub_mean(d):
    x,y = d
    x = x - np.mean(x,axis=0)
    return (x,y)

def std_norm(d):
    x,y = d
    x = x / np.std(x,axis=0)
    return(x,y)

def load_single_file(filename, resize_to = None, center_and_normalise = False,
                     one_hot_y = False, zca_whitening = False):
  ''' Loads the dataset

  :type dataset: string
  :param dataset: the path to the dataset (here MNIST)
  '''

  data = np.load(filename + '.npz')
  data = (data['x'],data['y'])
  
  #UNIFORM_PADDING
  if resize_to is not None:
    orig_size = math.sqrt(data[0].shape[1])
    data = (pad_dataset(
              data[0].reshape((data[0].shape[0], orig_size,orig_size)),
              resize_to),
            data[1])
  #MEANSTD
  if center_and_normalise:
    data = std_norm(sub_mean(data))

  #ZCA WHITENING
  if zca_whitening:
      print("WARNING: ZCA Whitening dataset, you will need the preprocessor to be able to run the network after training")
      data_shape = data[0].shape
      flat_shape = np.prod(data_shape[1:])
      data_flat = data[0].reshape((data_shape[0],flat_shape))
      print("finished reshaping")
      sigma = np.dot(data_flat.T, data_flat) / data_flat.shape[0] #Correlation matrix
      print("got correlation matrix")
      U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
      print("got SVD")
      epsilon = 0.1                #Whitening constant, it prevents division by zero
      ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
      print("got whitening matrix")
      def whiten(inputs):
        return np.dot(ZCAMatrix, inputs) 
      data_zca = whiten(data_flat).reshape(data_shape)
      data = (data_zca, data[1])
      print("Done")

  if one_hot_y:
      data = (data[0], one_hot(data[1]))
  return data

def load_data(dataset, resize_to = None, pickled = True,
              center_and_normalise = False, join_train_and_valid = False,
              one_hot_y = False, zca_whitening = False):
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
    print('loading data...')
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
  else:
    tr = np.load(dataset + 'train.npz')
    v = np.load(dataset + 'valid.npz')
    te = np.load(dataset + 'test.npz')
    train_set = (tr['x'],tr['y'])
    valid_set = (v['x'],v['y'])
    test_set = (te['x'],te['y'])
  
  #UNIFORM_PADDING
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
  #MEANSTD
  if center_and_normalise:
    train_set = std_norm(sub_mean(train_set))
    valid_set = std_norm(sub_mean(valid_set))
    test_set  = std_norm(sub_mean(test_set))

  #NO VALID (usually CIFAR)
  if join_train_and_valid:
    set_x = numpy.concatenate([
                train_set[0],
                valid_set[0]
            ])
    set_y = numpy.concatenate([
                train_set[1],
                valid_set[1]
            ])
    train_set = (set_x,set_y)
    valid_set = test_set

  #ZCA WHITENING
  if zca_whitening:
      print("WARNING: ZCA Whitening dataset, you will need the preprocessor to be able to run the network after training")
      train_shape = train_set[0].shape
      valid_shape = valid_set[0].shape
      test_shape = test_set[0].shape
      flat_shape = np.prod(train_shape[1:])
      train_flat = train_set[0].reshape((train_shape[0],flat_shape))
      valid_flat = valid_set[0].reshape((valid_shape[0],flat_shape))
      test_flat = test_set[0].reshape((test_shape[0],flat_shape))
      print("finished reshaping")
      sigma = np.dot(train_flat.T, train_flat) / train_flat.shape[0] #Correlation matrix
      print("got correlation matrix")
      U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
      print("got SVD")
      epsilon = 0.1                #Whitening constant, it prevents division by zero
      ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
      print("got whitening matrix")
      def whiten(inputs):
        return np.dot(ZCAMatrix, inputs) 
      train_zca = whiten(train_flat).reshape(train_shape)
      valid_zca = whiten(valid_flat).reshape(valid_shape)
      test_zca = whiten(test_flat).reshape(test_shape)
      train_set = (train_zca, train_set[1])
      valid_set = (valid_zca, valid_set[1])
      test_set = (test_zca, train_set[1])
      print("Done")

  if one_hot_y:
      train_set = (train_set[0], one_hot(train_set[1]))
      valid_set = (valid_set[0], one_hot(valid_set[1]))
      test_set = (test_set[0], one_hot(test_set[1]))
  return (train_set, valid_set, test_set)

def make_pretraining_set(datasets,mode):
  if mode is not None:
    return (datasets[0][0],datasets[0][1])
  else:
    return None


class Resampler:
    """
    Resample a dataset either uniformly or with a given probability
    distribution
    """

    def __init__(self, dataset, seed = 42):
        self.train,self.valid,self.test = dataset
        self.train_x, self.train_y = self.train
        self.valid_x, self.valid_y = self.valid
        self.test_x, self.test_y = self.test
        self.train_size = len(self.train_x)
        self.r_train = None
        np.random.seed(seed)
        
    def make_new_train(self,sample_size,distribution=None):
        weights = []
        if distribution is None:
            sample = np.random.randint(low=0,
                                       high=self.train_size,
                                       size=sample_size)
        else:
            values = (range(len(distribution)),distribution)
            d = scipy.stats.rv_discrete(a=0,b=len(distribution),values=values)
            sample = d.rvs(size=sample_size)
        sampled_x = []
        sampled_y = []
        for s in sample:
            sampled_x.append(self.train_x[s])
            sampled_y.append(self.train_y[s])
        if distribution is not None:
            for s in sample:
                weights.append(distribution[s])
            weights = numpy.asarray(weights)
        else:
            weights = None
        sampled_x = numpy.asarray(sampled_x)
        sampled_y = numpy.asarray(sampled_y)
        self.r_train = (sampled_x,sampled_y)
        return self.r_train, weights

    def get_train(self):
        return self.train

    def get_valid(self):
        return self.valid

    def get_test(self):
        return self.test


class WeightedResampler(Resampler):

    def __init__(self, dataset, seed = 42):
        Resampler.__init__(self, dataset, seed)
        self.weights = numpy.repeat([1.0/self.train_size], self.train_size)

    def update_weights(self,new_weights):
        self.weights = new_weights

    def make_new_train(self,sample_size):
        return Resampler.make_new_train(self,sample_size,self.weights)

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
        bp = int(round(padding / 2)) # before padding (left)
        ap = int(round(padding - bp)) # after padding (right)
        pads = (bp,ap)
        if bp + ap > 0:
            new_x.append(np.pad(x,(pads,pads),mode='constant').reshape(end_size**2))
        else: # image is too big now, unpad/slice
            si = -bp # start index
            ei = cs + ap # end index
            new_x.append(x[si:ei, si:ei].reshape(end_size**2))
    return np.asarray(new_x)

class Transformer:
    """
    Apply translation, scaling, rotation and other transformations to a 
    training set to produce a larger, noisy training set
    """

    def __init__(self,original_set,x,y,alpha,beta,gamma,sigma,noise_var,rng,progress = False):
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
        self.original_x = np.asarray(original_set)
        self.final_x = []
        self.instance_no = 0
        instances = len(self.original_x)
        self.original_x = np.asarray(self.original_x).reshape(instances,self.x,self.y)
        p = multiprocessing.Pool(32)
        deferred = [p.apply_async(transform_aux_map,args=(self,x)) for x in
                self.original_x]
        self.final_x = [x.get() for x in deferred]

    def apply(self,curr_x):
#        if self.progress and self.instance_no % 100 == 0:
#            print("instance {0}".format(self.instance_no), end="\r")
        dx = np.random.uniform(low=self.min_trans_x,high=self.max_trans_x)
        dy = np.random.uniform(low=self.min_trans_y,high=self.max_trans_y)
        curr_x = self.translate_instance(curr_x,int(dx),int(dy))
        angle = np.random.uniform(low=-self.beta,high=self.beta)
        shear = np.random.uniform(low=-self.beta,high=self.beta)
        curr_x = self.rotate_instance(curr_x,angle)
        #curr_x = self.gaussian_noise(curr_x,noise_var)
        scale_x = 1. + np.random.uniform(low=-self.gamma,high=self.gamma) / 100.
        scale_y = 1. + np.random.uniform(low=-self.gamma,high=self.gamma) / 100.
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
            field_x = np.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
            field_y = np.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
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
        return xval + np.random.normal(0.,sigma,xval.shape)

    def scale(self,xval,scaling):
        return ni.zoom(xval,scaling)

    def get_data(self):
        return np.array(self.final_x)

def one_hot(dataset):
    b = np.zeros((dataset.size, dataset.max()+1),dtype='float32')
    b[np.arange(dataset.size), dataset] = 1.
    return b
