#!/usr/bin/python


"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

#TODO: Resampler, WeightedResampler
#TODO: Polymorphism for different data formats

import os
#import gc
import sys
from pathlib import Path
import numpy as np

#import scipy.ndimage as ni
#import scipy.stats
#import numpy.random
#import gzip
#import pickle
#import math
#from skimage import transform
#import multiprocessing
import tensorflow as tf
from toupee.common import dict_map

KNOWN_DATA_TYPES = [".tfrecord", ".h5", ".npz"]

DEFAULT_TRAINING_FILE = 'train'
DEFAULT_VALIDATION_FILE = 'valid'
DEFAULT_TESTING_FILE = 'test'

def get_data_format(filename):
    """ Identifies the extension of the dataset """
    extension = Path(filename).suffix
    if extension not in KNOWN_DATA_TYPES:
        raise ValueError("Unknown data type %s" % extension)
    return extension

def one_hot_numpy(dataset):
    b = np.zeros((dataset.size, dataset.max()+1),dtype='float32')
    b[np.arange(dataset.size), dataset] = 1.
    return b

def _load_h5(filename, **kwargs):
    """ Load an HDF5 file """
    #TODO: write me
    raise NotImplementedError()


def _load_npz(filename, **kwargs):
    """ Load a NPZ file """
    #TODO: transformations
    #TODO: special dict mappings
    data = np.load(filename)
    data = (data['x'],data['y'])
    if kwargs['convert_to_one_hot_y']:
        data = (data[0], one_hot_numpy(data[1]))
    return data


def _load_tfrecord(filename, **kwargs):
    """ Load a TFRecord file """
    #TODO: write me
    raise NotImplementedError()


def load(filename, **kwargs):
    """ Load any known data format """
    mapper = {'.h5': _load_h5,
              '.npz': _load_npz,
              '.tfrecord': _load_tfrecord
             }
    return mapper[get_data_format(filename)](filename, **kwargs)


def _np_to_tf(data, batch_size, **kwargs):
    """ Convert an np dataset to a tfrecord """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    return dataset


def convert_to_tf(data, data_format, **kwargs):
    """ Convert current data to tf """
    mapper = {'.h5': None,
              '.npz': _np_to_tf,
              '.tfrecord': None
             }
    return mapper[data_format](data, **kwargs)


def _resample_np(data, sample_size, weights, replace):
    """ Rasampling specifically for np arrays """
    data_size = data[0].shape[0]
    indices = np.random.choice(range(data_size), size=sample_size, replace=replace, p=weights)
    return tuple(np.take(feature, indices, axis=0) for feature in data)


class Dataset:
    """ Class to load a dataset """
    def __init__(self,
                 src_dir=None,
                 training_file=None,
                 validation_file=None,
                 testing_file=None,
                 data_format="",
                 **kwargs):
        self.files = {}
        if src_dir is None and training_file is None:
            raise ValueError("Must specify one of src_dir or training_file")
        self.files['train'] = training_file or DEFAULT_TRAINING_FILE
        self.files['valid'] = validation_file or DEFAULT_VALIDATION_FILE
        self.files['test'] = testing_file or DEFAULT_TESTING_FILE
        if src_dir:
            self.files = dict_map(self.files, lambda f_name: os.path.join(src_dir, f_name))
        if data_format:
            self.files = dict_map(self.files, lambda f_name: f_name + '.' + data_format)
        if not os.path.exists(self.files['train']):
            raise ValueError('Training file %s not found!' % self.files['train'])
        self.files = dict_map(self.files, lambda f_name: f_name if os.path.exists(f_name) else None)
        self.data_format = get_data_format(self.files['train'])
        for f_name in self.files.values():
            if f_name is not None and get_data_format(f_name) != self.data_format:
                raise ValueError("All files must be in same format")
        self.raw_data = dict_map(self.files, lambda f_name: load(filename=f_name, **kwargs) if f_name else None)
        self.data = dict_map(self.raw_data, lambda data: convert_to_tf(data=data, data_format=self.data_format, **kwargs))
        self.kwargs = kwargs

    def get_training_handle(self):
        """ Return the appropriate handle to pass to keras .fit """
        return self.data['train']
    
    def get_testing_handle(self):
        """ Return the appropriate handle to pass to keras .evaluate """
        return self.data['test']

    def resample(self, sample_size=None, weights=None, replace=True):
        """ Promote the dataset to a resampling one """
        self.resample_size = sample_size or self.raw_data['train'][0].shape[0]
        self.resample_weights = weights
        self.resample_replace = replace
        self.__class__ = ResamplingDatasetWrapper
        return self


class ResamplingDatasetWrapper(Dataset):
    """ A dataset that resamples every time a handle is given """

    def __init__(self, **kwargs):
        raise RuntimeError("ResamplingDatasetWrapper cannot be created directly, use Dataset.resample")

    def get_training_handle(self):
        print("!!! RESAMPLING")
        mapper = {'.h5': None,
                  '.npz': _resample_np,
                  '.tfrecord': None
            }
        resampled = mapper[self.data_format](self.raw_data['train'],
                                             sample_size=self.resample_size,
                                             weights=self.resample_weights,
                                             replace=self.resample_replace)
        return convert_to_tf(resampled, self.data_format, **self.kwargs)


#### THIS IS ALL OLD

# def corrupt(data,p):
#     return mask(1-p,data.shape,dtype=floatX) * data

# def sub_mean(d):
#     x,y = d
#     x = x - np.mean(x,axis=0)
#     return (x,y)

# def std_norm(d):
#     x,y = d
#     x = x / np.std(x,axis=0)
#     return(x,y)

# def load_single_file(filename, resize_to = None, center_and_normalise = False,
#                      one_hot_y = False, zca_whitening = False):
#   ''' Loads the dataset

#   :type dataset: string
#   :param dataset: the path to the dataset (here MNIST)
#   '''
#   raise DeprecationWarning("Use TF.Dataset")
#   data = np.load(filename + '.npz')
#   data = (data['x'],data['y'])
  
#   #UNIFORM_PADDING
#   if resize_to is not None:
#     orig_size = math.sqrt(data[0].shape[1])
#     data = (pad_dataset(
#               data[0].reshape((data[0].shape[0], orig_size,orig_size)),
#               resize_to),
#             data[1])
#   #MEANSTD
#   if center_and_normalise:
#     data = std_norm(sub_mean(data))

#   #ZCA WHITENING
#   if zca_whitening:
#       print("WARNING: ZCA Whitening dataset, you will need the preprocessor to be able to run the network after training")
#       data_shape = data[0].shape
#       flat_shape = np.prod(data_shape[1:])
#       data_flat = data[0].reshape((data_shape[0],flat_shape))
#       print("finished reshaping")
#       sigma = np.dot(data_flat.T, data_flat) / data_flat.shape[0] #Correlation matrix
#       print("got correlation matrix")
#       U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
#       print("got SVD")
#       epsilon = 0.1                #Whitening constant, it prevents division by zero
#       ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
#       print("got whitening matrix")
#       def whiten(inputs):
#         return np.dot(ZCAMatrix, inputs) 
#       data_zca = whiten(data_flat).reshape(data_shape)
#       data = (data_zca, data[1])
#       print("Done")

#   if one_hot_y:
#       data = (data[0], one_hot(data[1]))
#   return data

# def load_data(dataset, resize_to = None, pickled = True,
#               center_and_normalise = False, join_train_and_valid = False,
#               one_hot_y = False, zca_whitening = False,
#               trainfile  = 'train.npz', validfile = 'valid.npz', testfile = 'test.npz'):
#   ''' Loads the dataset

#   :type dataset: string
#   :param dataset: the path to the dataset (.npz)
#   '''

#   data_dir, data_file = os.path.split(dataset)
#   if pickled:
#     if data_dir == "" and not os.path.isfile(dataset):
#       new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
#       if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
#         dataset = new_path
#     print('loading data...')
#     f = gzip.open(dataset, 'rb')
#     train_set, valid_set, test_set = pickle.load(f)
#     f.close()
#   else:
#     #loads the .npz
#     if (trainfile[-4:] == '.npz') and (validfile[-4:] == '.npz') and (testfile[-4:] == '.npz'):
#         tr = np.load(os.path.join(dataset, trainfile))
#         v = np.load(os.path.join(dataset, validfile))
#         te = np.load(os.path.join(dataset, testfile))
#     else:
#         raise ValueError('.npz files are required here; All sets must have the same format.')

#     if 'x' in tr and 'x' in v and 'x' in te:
#         xlabel = 'x'
#     elif 'X' in tr and 'X' in v and 'X' in te:
#         xlabel = 'X'
#     else:
#         raise ValueError('training, testing and validation sets must have either x or X fields')
#     train_set = (tr[xlabel],tr['y'])
#     valid_set = (v[xlabel],v['y'])
#     test_set = (te[xlabel],te['y'])

#   #UNIFORM_PADDING
#   if resize_to is not None:
#     orig_size = math.sqrt(train_set[0].shape[1])
#     train_set = (
#                   pad_dataset(
#                       train_set[0].reshape((train_set[0].shape[0],
#                                               orig_size,orig_size)
#                       ),
#                       resize_to),
#                   train_set[1])
#     valid_set = (
#                   pad_dataset(
#                       valid_set[0].reshape((valid_set[0].shape[0],
#                                               orig_size,orig_size)
#                       ),
#                       resize_to),
#                   valid_set[1])
#     test_set  = (
#                   pad_dataset(
#                       test_set[0].reshape((test_set[0].shape[0],
#                                               orig_size,orig_size)
#                       ),
#                       resize_to),
#                   test_set[1])
#   #MEANSTD
#   if center_and_normalise:
#     train_set = std_norm(sub_mean(train_set))
#     valid_set = std_norm(sub_mean(valid_set))
#     test_set  = std_norm(sub_mean(test_set))

#   #NO VALID (usually CIFAR)
#   if join_train_and_valid:
#     set_x = numpy.concatenate([
#                 train_set[0],
#                 valid_set[0]
#             ])
#     set_y = numpy.concatenate([
#                 train_set[1],
#                 valid_set[1]
#             ])
#     train_set = (set_x,set_y)
#     valid_set = test_set

#   #ZCA WHITENING
#   if zca_whitening:
#       print("WARNING: ZCA Whitening dataset, you will need the preprocessor to be able to run the network after training")
#       train_shape = train_set[0].shape
#       valid_shape = valid_set[0].shape
#       test_shape = test_set[0].shape
#       flat_shape = np.prod(train_shape[1:])
#       train_flat = train_set[0].reshape((train_shape[0],flat_shape))
#       valid_flat = valid_set[0].reshape((valid_shape[0],flat_shape))
#       test_flat = test_set[0].reshape((test_shape[0],flat_shape))
#       print("finished reshaping")
#       sigma = np.dot(train_flat.T, train_flat) / train_flat.shape[0] #Correlation matrix
#       print("got correlation matrix")
#       U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
#       print("got SVD")
#       epsilon = 0.1                #Whitening constant, it prevents division by zero
#       ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
#       print("got whitening matrix")
#       def whiten(inputs):
#         return np.dot(ZCAMatrix, inputs) 
#       train_zca = whiten(train_flat).reshape(train_shape)
#       valid_zca = whiten(valid_flat).reshape(valid_shape)
#       test_zca = whiten(test_flat).reshape(test_shape)
#       train_set = (train_zca, train_set[1])
#       valid_set = (valid_zca, valid_set[1])
#       test_set = (test_zca, train_set[1])
#       print("Done")

#   if one_hot_y:
#       train_set = (train_set[0], one_hot(train_set[1]))
#       valid_set = (valid_set[0], one_hot(valid_set[1]))
#       test_set = (test_set[0], one_hot(test_set[1]))
#   return (train_set, valid_set, test_set)

# def make_pretraining_set(datasets,mode):
#   if mode is not None:
#     return (datasets[0][0],datasets[0][1])
#   else:
#     return None

    
# class Resampler:
#     """
#     Resample a dataset either uniformly or with a given probability
#     distribution
#     """

#     def __init__(self, train_size, seed = 42):
        
#         self.r_train = None
#         np.random.seed(seed)
#         self.train_size = train_size
        
#     def make_new_train(self,sample_size,distribution=None):
        
#         #with returns the indexes, not the samples themselves
        
#         weights = []
        
#         #gets the sample indexes
#         if distribution is None:
#             sample = np.random.randint(low=0,
#                                        high=self.train_size,
#                                        size=sample_size)
#         else:
#             sample = np.random.choice(len(distribution), size = sample_size, 
#                                         p = distribution)
            
#         #sets the selected weights
#         if distribution is not None:
#             for s in sample:
#                 weights.append(distribution[s])
#             weights = numpy.asarray(weights)
#         else:
#             weights = None
         
#         self.r_train = sample
       
#         #returns the indexes/samples, depending on the case
#         return self.r_train, weights


# class WeightedResampler(Resampler):

#     def __init__(self, train_size, seed = 42):
#         Resampler.__init__(self, train_size, seed = seed)
#         self.weights = numpy.repeat([1.0/self.train_size], self.train_size)

#     def update_weights(self,new_weights):
#         self.weights = new_weights

#     def make_new_train(self,sample_size):
#         return Resampler.make_new_train(self,sample_size,self.weights)

# def transform_aux_map(tr,x):
#     return tr.apply(x)

# def f(x):
#     return x

# def pad_dataset(xval,end_size):
#     """ 
#     Thanks to https://github.com/ilyakava/ciresan
#     """
#     new_x = []
#     for x in xval:
#         cs = x.shape[0]
#         padding = end_size - cs
#         bp = int(round(padding / 2)) # before padding (left)
#         ap = int(round(padding - bp)) # after padding (right)
#         pads = (bp,ap)
#         if bp + ap > 0:
#             new_x.append(np.pad(x,(pads,pads),mode='constant').reshape(end_size**2))
#         else: # image is too big now, unpad/slice
#             si = -bp # start index
#             ei = cs + ap # end index
#             new_x.append(x[si:ei, si:ei].reshape(end_size**2))
#     return np.asarray(new_x)

# class Transformer:
#     """
#     Apply translation, scaling, rotation and other transformations to a 
#     training set to produce a larger, noisy training set
#     """

#     def __init__(self,original_set,x,y,alpha,beta,gamma,sigma,noise_var,rng,progress = False):
#         print("..transforming dataset")
#         self.progress = progress
#         self.x = x
#         self.y = y
#         self.min_trans_x = -2
#         self.max_trans_x =  2
#         self.min_trans_y = -2
#         self.max_trans_y =  2
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.sigma = sigma
#         self.noise_var = noise_var
#         self.original_x = np.asarray(original_set)
#         self.final_x = []
#         self.instance_no = 0
#         instances = len(self.original_x)
#         self.original_x = np.asarray(self.original_x).reshape(instances,self.x,self.y)
#         p = multiprocessing.Pool(32)
#         deferred = [p.apply_async(transform_aux_map,args=(self,x)) for x in
#                 self.original_x]
#         self.final_x = [x.get() for x in deferred]

#     def apply(self,curr_x):
# #        if self.progress and self.instance_no % 100 == 0:
# #            print("instance {0}".format(self.instance_no), end="\r")
#         dx = np.random.uniform(low=self.min_trans_x,high=self.max_trans_x)
#         dy = np.random.uniform(low=self.min_trans_y,high=self.max_trans_y)
#         curr_x = self.translate_instance(curr_x,int(dx),int(dy))
#         angle = np.random.uniform(low=-self.beta,high=self.beta)
#         shear = np.random.uniform(low=-self.beta,high=self.beta)
#         curr_x = self.rotate_instance(curr_x,angle)
#         #curr_x = self.gaussian_noise(curr_x,noise_var)
#         scale_x = 1. + np.random.uniform(low=-self.gamma,high=self.gamma) / 100.
#         scale_y = 1. + np.random.uniform(low=-self.gamma,high=self.gamma) / 100.
#         curr_x = self.scale(curr_x,[scale_x,scale_y])
# #            curr_x = self.elastic_transform(curr_x,sigma,alpha)
# #        trans = tramsform.AffineTransform(
# #                    scale=(scale_x,scale_y),
# #                    shear=shear,
# #                    rotation=angle,
# #                    translation=(dx,dy)
# #                )
# #        warped_x = transform.warp(curr_x,trans)
# #        if plot:
# #            plt.gray()
# #            plt.imshow(warped_x)
# #            plt.show()
# #        self.instance_no += 1
#         warped_x = curr_x
#         return warped_x.flatten()

#     def elastic_transform(self,xval,sigma,alpha):
#             field_x = np.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
#             field_y = np.random.rand(xval.shape[0],xval.shape[1]) * 2. - 1.
#             convolved_field_x = ni.filters.gaussian_filter(field_x,sigma)
#             convolved_field_y = ni.filters.gaussian_filter(field_y,sigma)
#             convolved_field_x = convolved_field_x * alpha / max(abs(convolved_field_x.flatten()))
#             convolved_field_y = convolved_field_y * alpha / max(abs(convolved_field_y.flatten()))
#             def mapping(coords):
#                 x,y = coords
#                 return  (x+convolved_field_x[x,y],y+convolved_field_y[x,y])
#             return ni.geometric_transform(xval,mapping)

#     def translate_instance(self,xval, dx, dy):
#         return np.roll(np.roll(xval,dx,axis=0),dy,axis=1)

#     def rotate_instance(self,xval,angle):
#         return transform.rotate(xval,angle, mode='reflect')

#     def gaussian_noise(self,xval,sigma):
#         return xval + np.random.normal(0.,sigma,xval.shape)

#     def scale(self,xval,scaling):
#         return ni.zoom(xval,scaling)

#     def get_data(self):
#         return np.array(self.final_x)

# def one_hot(dataset):
#     b = np.zeros((dataset.size, dataset.max()+1),dtype='float32')
#     b[np.arange(dataset.size), dataset] = 1.
#     return b
