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

from keras.callbacks import Callback
from keras.utils import Sequence

numpy.set_printoptions(threshold=numpy.inf)


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
                          


#Joao: I tried to prefetch the data from the disk in this generator, but it led to
#       multiple complications (especially with resampled data). 
#       But it can decrease train&test time - do it in the future!
class DataGenerator(Sequence):
    ''' 
        Data holder generator class for .npz/.h5 data 
        -- keras Sequence based (for better generator performance)
            [requires __len__(self) and __getitem__(self, idx)]
    '''
    
    def __init__(self, data_file, batch_size, sampled_indexes, hold_y = True):
        
        #define x
        if 'x' in data_file:
            xlabel = 'x'
        elif 'X' in data_file:
            xlabel = 'X'
        
        self.data_x = data_file[xlabel]
        

        #auxiliary variables
        self.sampled_indexes = sampled_indexes
        if sampled_indexes is not None:
            self.num_examples = len(sampled_indexes)
        else:
            self.num_examples = self.data_x.shape[0]
        self.batch_size = batch_size
        self.number_of_batches = math.ceil(self.num_examples/self.batch_size)
    
    
        # define y if needed
        self.hold_y = hold_y
        if hold_y:
            #Todo: classification problem  -for now it assumes that 
            #       y is a one-hot thing, which might not be always true!
            self.n_classes = data_file['y'].shape[1]
            assert self.n_classes > 1
            self.data_y = data_file['y']

    
    def sequential_batch(self, step):
        #sequential iteration over the data
        
        #defines the indexes for this batch
        if (step+1) == self.number_of_batches:    #<- last batch
            batch_indexes = list(range(step*self.batch_size, self.num_examples))
        else:
            batch_indexes = list(range(step*self.batch_size, (step+1)*self.batch_size))
    
    
        if self.hold_y:
            # Return the arrays in the shape that fit_gen uses (data, target)
            return (self.data_x[batch_indexes, ...],
                    self.data_y[batch_indexes, ...])
        # else:
        # Return the arrays in the shape that predict_generator uses (data)
        return (self.data_x[batch_indexes, ...]) 
            
            
    def sliced_batch(self, step):
        #problem with returning the "sampled_indexes" only:
        # H5 can only slice given i) a sequencial list of integers or ii) a boolean array
        # [i.e. there is no fancy slicing, as in numpy]
        # since ii) might need a giant boolean array, let's do i) and then filter stuff
        
        #gets the desired indexes for this batch
        if (step+1) == self.number_of_batches:    #<- last batch
            batch_indexes = (self.sampled_indexes[step*self.batch_size : self.num_examples])
        else:
            batch_indexes = (self.sampled_indexes[step*self.batch_size : (step+1)*self.batch_size])
         
        first_index = batch_indexes[0]
        last_index = batch_indexes[-1]
        
        #if the samples are too far appart, loads one by one
        if last_index - first_index > 4096:   #<---- magic number
            
            data_x = []
            for i in batch_indexes:
                data_x.append(self.data_x[i, ...])  
            data_x = numpy.asarray(data_x)
            
            if self.hold_y:
                data_y = []
                for i in batch_indexes:
                    data_y.append(self.data_y[i, ...])  
                data_y = numpy.asarray(data_y)

        #otherwise, loads the interval and then filters
        else:
            batch_indexes = batch_indexes - first_index
            
            data_x = self.data_x[first_index:last_index+1, ...]
            data_x = data_x[batch_indexes, ...]
            
            if self.hold_y:
                data_y = self.data_y[first_index:last_index+1, ...]
                data_y = data_y[batch_indexes, ...]
                
            
        if self.hold_y:    
            return(data_x, data_y)
            
        # else:
        return(data_x)
    
    
    def __len__(self):
        #returns the dataset length
        return self.number_of_batches
    
    
    def __getitem__(self, step):
        #gets a batch
        if self.sampled_indexes is None:
            return self.sequential_batch(step)
        else:
            return self.sliced_batch(step)
            




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

    with file(file_name) as inputfile:
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


if 'toupee_global_instance' not in locals():
    toupee_global_instance = Toupee()

                
#for classification problems: 

#TODO: pass as argument the number of classes               
def errors(classifier, file_object, batch_size):

    x_holder = DataGenerator(file_object, batch_size, None, hold_y = False)
    
    #applies the correct method, depending on the classifier class
    if hasattr(classifier, 'predict_generator'):
        classification_proba = classifier.predict_generator(x_holder,
                                                            max_queue_size=1000)
    else:
        classification_proba = classifier.predict_proba(x_holder)

    
    #converts to the predicted class (integer)
    if classification_proba.shape[-1] > 1:
        classification = classification_proba.argmax(axis=-1)
    else:
        classification = (classification_proba > 0.5).astype('int32')

    #TODO: if y is a one-hot vector, changes are needed!
    #gets the result (iterativelly, to avoid running out of memory)
    end = 0
    r = numpy.empty(x_holder.num_examples)
    while end < (x_holder.__len__() * batch_size):
        
        start = end
        end += 131072  # magic number, power of 2 :D
        if end > x_holder.__len__() * batch_size:
            end = x_holder.__len__() * batch_size
        
        data_y = numpy.asarray(file_object['y'][start:end]).argmax(axis=-1)
        r[start:end] = (classification[start:end] - data_y).astype(bool).astype('int32')

    return r
    
    
def accuracy(classifier, file_object, batch_size):
    
    e = errors(classifier, file_object, batch_size)
    
    return 1.0 - (float(e.sum()) / float(file_object['y'].shape[0]))
 
 
#TODO: this is kinda a redefinition of data.py's one_hot -> take care of the duplicates!
def one_hot(data, n_classes):
    b = numpy.zeros((data.size, n_classes),dtype='float32')
    b[numpy.arange(data.size), data] = 1.
    return b
    

    
    
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
    