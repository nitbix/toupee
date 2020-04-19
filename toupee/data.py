#!/usr/bin/python


"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

#TODO: Polymorphism for different data formats
#TODO: Online image transformations

import os
import sys
import math

from pathlib import Path
import numpy as np

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
    """ Convert a numpy array of Y labels into one-hot encodings """
    n_classes = dataset.max() + 1
    return np.eye(n_classes)[dataset.reshape(-1)]

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
    if kwargs['convert_labels_to_one_hot']:
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


def _np_to_tf(data, batch_size, shuffle=False, shuffle_buffer=None, gen_flow=None, **kwargs):
    """ Convert an np dataset to a tfrecord """
    if gen_flow:
        dataset = tf.data.Dataset.from_generator(
            lambda: gen_flow,
            #img_gen.flow, args=[data[0], data[1], batch_size, shuffle], #TODO: figure out how to make them named
            output_types=(tf.float32, tf.float32),
            output_shapes=((None,) + data[0].shape[1:],
                           (None,) + data[1].shape[1:])
        )
        dataset = dataset.unbatch()
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    if shuffle and not gen_flow: # gen_flow shuffle by itself and this will make things super slow
        buffer = shuffle_buffer or data[0].shape[0]
        dataset = dataset.shuffle(buffer)
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
                 shuffle=False,
                 img_gen_params=None,
                 max_examples=None,
                 **kwargs):
        self.files = {}
        self.shuffle = shuffle
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
        self.img_gen_params = img_gen_params
        for f_name in self.files.values():
            if f_name is not None and get_data_format(f_name) != self.data_format:
                raise ValueError("All files must be in same format")
        self.raw_data = dict_map(self.files,
                                 lambda f_name: load(filename=f_name, **kwargs) if f_name else None)
        self.n_classes = self.raw_data['train'][1].shape[1]
        if max_examples:
            self.raw_data = dict_map(self.raw_data, lambda d: (d[0][:max_examples], d[1][:max_examples]))
        self.size = dict_map(self.raw_data, lambda data: data[0].shape[0])
        self.steps_per_epoch = dict_map(self.size,
                                        lambda size: math.ceil(float(size) / kwargs['batch_size']))
        if self.img_gen_params:
            self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(**self.img_gen_params)
            self.img_gen.fit(self.raw_data['train'][0])
            self.train_flow = self.img_gen.flow(self.raw_data['train'][0],
                                                self.raw_data['train'][1],
                                                batch_size=kwargs['batch_size'],
                                                shuffle=self.shuffle)
            # Only standardize validation set because the test set is standardized in predict_proba
            self.standardized_raw_data = dict_map(self.raw_data,
                                                  lambda data: (self.img_gen.standardize(np.copy(data[0])),
                                                                data[1]))
            self.standardized_data = dict_map(
                {k: self.standardized_raw_data[k] for k in  ('valid', 'test')},
                lambda data: convert_to_tf(data=data,
                                            data_format=self.data_format,
                                            **kwargs))
        self.data = dict_map({k: self.raw_data[k] for k in  ('valid', 'test')},
                             lambda data: convert_to_tf(data=data,
                                                        data_format=self.data_format,
                                                        **kwargs))
        self.data['train'] = convert_to_tf(data=self.raw_data['train'],
                                           data_format=self.data_format,
                                           shuffle=self.shuffle,
                                           gen_flow=self.train_flow,
                                           **kwargs)
        self.kwargs = kwargs

    def get_training_handle(self):
        """ Return the appropriate handle to pass to keras .fit """
        return self.data['train']

    def get_handle(self, split, standardized=False):
        """ Return the appropriate handle to pass to keras .fit as validation_data """
        if standardized:
            return self.standardized_data[split]
        return self.data[split]

    def get_validation_handle(self, standardized=False):
        """ Return the appropriate handle to pass to keras .fit as validation_data """
        return self.get_handle(split='valid', standardized=standardized)

    def get_testing_handle(self, standardized=False):
        """ Return the appropriate handle to pass to keras .evaluate """
        return self.get_handle(split='test', standardized=standardized)

    def resample(self, sample_size=None, weights=None, replace=True):
        """ Promote the dataset to a resampling one """
        self.resample_size = sample_size or self.raw_data['train'][0].shape[0]
        self.resample_weights = weights
        self.resample_replace = replace
        self.__class__ = ResamplingDataset
        return self


class ResamplingDataset(Dataset):
    """ A dataset that resamples every time a training handle is requested """

    def __init__(self, **kwargs):
        raise RuntimeError("ResamplingDatasetWrapper cannot be created directly, use Dataset.resample")

    def set_weights(self, weights):
        """ Change the resampling weights """
        self.resample_weights = weights

    def get_training_handle(self):
        """ Returns training tf.dataset, resampled every time from raw_data """
        mapper = {'.h5': None,
                  '.npz': _resample_np,
                  '.tfrecord': None
            }
        resampled = mapper[self.data_format](self.raw_data['train'],
                                             sample_size=self.resample_size,
                                             weights=self.resample_weights,
                                             replace=self.resample_replace)
        if self.img_gen:
            #TODO: use the sample weights here?
            resampled_flow = self.img_gen.flow(resampled[0],
                                          resampled[1],
                                          batch_size=self.kwargs['batch_size'])
        return convert_to_tf(resampled, self.data_format, shuffle=self.shuffle, gen_flow=resampled_flow, **self.kwargs)