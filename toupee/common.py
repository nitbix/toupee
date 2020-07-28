#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy
import yaml
import os
import collections
import math
import time
import h5py
import warnings

def dict_map(dictionary, f):
    return {k: f(v) for k, v in dictionary.items()}


#KERAS ADD-ON
# class ModelCheckpointInMemory(Callback):
#     '''Save the model after every epoch in memory.
#     # Arguments
#         monitor: quantity to monitor.
#         verbose: verbosity mode, 0 or 1.
#         mode: one of {auto, min, max}.
#             If `save_best_only=True`, the decision
#             to overwrite the current save file is made
#             based on either the maximization or the
#             minimization of the monitored quantity. For `val_acc`,
#             this should be `max`, for `val_loss` this should
#             be `min`, etc. In `auto` mode, the direction is
#             automatically inferred from the name of the monitored quantity.
#     '''
#     def __init__(self, monitor='val_loss', verbose=0, mode='auto'):
#         super(ModelCheckpointInMemory, self).__init__()
#         self.monitor = monitor
#         self.verbose = verbose
#         self.best_model = h5py.File("/dev/null", driver = 'core',
#                 backing_store = False)
#         self.best_epoch = None

#         if mode not in ['auto', 'min', 'max']:
#             warnings.warn('ModelCheckpoint mode %s is unknown, '
#                           'fallback to auto mode.' % (mode),
#                           RuntimeWarning)
#             mode = 'auto'

#         if mode == 'min':
#             self.monitor_op = numpy.less
#             self.best = numpy.Inf
#         elif mode == 'max':
#             self.monitor_op = numpy.greater
#             self.best = -numpy.Inf
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = numpy.greater
#                 self.best = -numpy.Inf
#             else:
#                 self.monitor_op = numpy.less
#                 self.best = numpy.Inf

#     def on_epoch_end(self, epoch, logs={}):
#         current = logs.get(self.monitor)
#         if current is None:
#             warnings.warn('Can save best model only with %s available, '
#                           'skipping.' % (self.monitor), RuntimeWarning)
#         else:
#             if self.monitor_op(current, self.best):
#                 if self.verbose > 0:
#                     print('Epoch %05d: %s improved from %0.5f to %0.5f,'
#                           ' saving model'
#                           % (epoch, self.monitor, self.best, current))
#                 self.best = current
#                 self.best_model = self.model.get_weights()
#                 self.best_epoch = epoch
#             else:
#                 if self.verbose > 0:
#                     print('Epoch %05d: %s did not improve' %
#                           (epoch, self.monitor))

# def yaml_include(loader, node):
#     # Get the path out of the yaml file
#     file_name = os.path.join(os.path.dirname(loader.name), node.value)

#     with file(file_name) as inputfile:
#         return yaml.load(inputfile)

# yaml.add_constructor("!include", yaml_include)

# def read_yaml_file(filename):
#     with open(filename, 'r') as f:
#         model_yaml = yaml.load(f)
#     return yaml.dump(model_yaml)

# def serialize(o):
#     if isinstance(o, numpy.float32):
#         return float(o)
#     else:
#         try:
#             return numpy.asfarray(o).tolist()
#         except:
#             if isinstance(o, object):
#                 if 'serialize' in dir(o) and isinstance(getattr(o,'serialize'), 
#                         collections.Callable):
#                     return o.serialize()
#                 if 'tolist' in dir(o) and isinstance(getattr(o,'tolist'), 
#                         collections.Callable):
#                     return o.tolist()
#                 try:
#                     return json.loads(json.dumps(o.__dict__,default=serialize))
#                 except:
#                     return str(o)
#             else:
#                 raise Exception("don't know how to save {0}".format(type(o)))


# def confidence(classifier, file_object, batch_size):
#     """
#     Returns the model's confidence for the true label
#     """
#     class_proba = get_probabilities(classifier, file_object, batch_size)
#     n_samples = file_object['y'].shape[0]
#     end = 0
#     h = numpy.empty(n_samples)
#     while end < n_samples:
#         start = end
#         end += 131072  # magic number, power of 2 :D
#         if end > n_samples:
#             end = n_samples
#         data_y = numpy.asarray(file_object['y'][start:end]).argmax(axis=-1)
#         for i in range(end - start):
#             h[start + i] = class_proba[start + i][data_y[i]]
#     return h

