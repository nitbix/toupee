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
import h5py

#numpy.set_printoptions(threshold=numpy.inf)

from keras.callbacks import Callback
from keras.utils import Sequence


#KERAS ADD-ON
class ModelCheckpointInMemory(Callback):
    '''Save the model after every epoch in memory.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
    '''
    def __init__(self, monitor='val_loss', verbose=0, mode='auto'):
        super(ModelCheckpointInMemory, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        
        self.best_model = h5py.File("/dev/null", driver = 'core',
                backing_store = False)
        self.best_epoch = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = numpy.less
            self.best = numpy.Inf
        elif mode == 'max':
            self.monitor_op = numpy.greater
            self.best = -numpy.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = numpy.greater
                self.best = -numpy.Inf
            else:
                self.monitor_op = numpy.less
                self.best = numpy.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model'
                          % (epoch, self.monitor, self.best, current))
                self.best = current
                self.best_model = self.model.get_weights()
                self.best_epoch = epoch
            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' %
                          (epoch, self.monitor))
                          
class Toupee:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_hooks = []
        self.reset_hooks = []

    def add_epoch_hook(self,hook):
        if hook not in self.epoch_hooks:
            self.epoch_hooks.append(hook)

    def add_reset_hook(self,hook):
        if hook not in self.reset_hooks:
            self.reset_hooks.append(hook)

class Results:

    def __init__(self,params):
        self.params = params

    def set_history(self,hist):
        self.history = hist.__dict__

    def set_final_observation(self,valid,test,epoch):
        self.best_valid = valid
        if test is not None:
            self.best_test = test
        self.best_epoch = epoch
        self.params = self.params.serialize()

class ConfiguredObject(yaml.YAMLObject):

    def _default_value(self, param_name, value):
        if param_name not in self.__dict__:
            self.__dict__[param_name] = value

def yaml_include(loader, node):
    # Get the path out of the yaml file
    file_name = os.path.join(os.path.dirname(loader.name), node.value)

    with open(file_name) as inputfile:
        return yaml.load(inputfile)

yaml.add_constructor("!include", yaml_include)

def read_yaml_file(filename):
    with open(filename, 'r') as f:
        model_yaml = yaml.load(f)
    return yaml.dump(model_yaml)

def serialize(o):
    if isinstance(o, numpy.float32):
        return float(o)
    else:
        try:
            return numpy.asfarray(o).tolist()
        except:
            if isinstance(o, object):
                if 'serialize' in dir(o) and isinstance(getattr(o,'serialize'), collections.Callable):
                    return o.serialize()
                if 'tolist' in dir(o) and isinstance(getattr(o,'tolist'), collections.Callable):
                    return o.tolist()
                try:
                    return json.loads(json.dumps(o.__dict__,default=serialize))
                except:
                    return str(o)
            else:
                raise Exception("don't know how to save {0}".format(type(o)))

                
#for classification problems:                
def errors(classifier, test_set_x, test_set_y):
    classification = classifier.predict_classes(test_set_x)
    c = numpy.argmax(test_set_y, axis=1)
    r = numpy.where(classification != c, 1.0, 0.0)
    return r

def accuracy(classifier, test_set_x, test_set_y):
    e = errors(classifier, test_set_x, test_set_y)
    return 1.0 - (float(e.sum()) / float(test_set_y.shape[0]))
    
    
    
#for regression problems:   
def distance(predictor, test_set_x, test_set_y):
    #returns the distance squared (=MMSE)
    prediction = predictor.predict(test_set_x)
    elementwise_d_squared = numpy.square(prediction - test_set_y)
    euclidian_distance_squared = numpy.sum(elementwise_d_squared, axis = 1)
    return euclidian_distance_squared

def euclidian_distance(predictor, test_set_x, test_set_y):
    #euclidian_distance = sqrt{(y[0]-y_pred[0])^2 + (y[1]-y_pred[1])^2 + ... + (y[n-1]-y_pred[n-1])^2}
    euclidian_distance_squared = distance(predictor, test_set_x, test_set_y)
    euclidian_distance = numpy.sqrt(euclidian_distance_squared)
    return(numpy.sum(euclidian_distance) / float(test_set_y.shape[0]))
    
def relative_distance(predictor, test_set_x, test_set_y):
    #relative_distance = distance(y-y_pred) / sqrt(y^2)      [sqrt(y^2) = L2 norm]
    euclidian_distance_squared = distance(predictor, test_set_x, test_set_y)
    y_squared = numpy.sum(numpy.square(test_set_y), axis = 1)                   #both this and the previous line will need a sqrt, which can be done after the division
    relative_distance = numpy.sqrt(euclidian_distance_squared / y_squared)
    return(numpy.sum(relative_distance) / float(test_set_y.shape[0]))
    
    
    

if 'toupee_global_instance' not in locals():
    toupee_global_instance = Toupee()
