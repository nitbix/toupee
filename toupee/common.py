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
import numpy as np

numpy.set_printoptions(threshold=numpy.inf)

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

#---------------------------------------------------------------------------------
# h5 stuff 
    
#TODO: implement shuffle for this data holder
class DataHolderH5:
    ''' Data holder "generator" [currently it's an iterator, got it's good enough] class for h5 data'''
    def __init__(self, file, batch_size, sampled_indexes, hold_y = True, to_one_hot = False):
        self.data_x = file['x']
        
        self.hold_y = hold_y
        self.to_one_hot = to_one_hot
        if hold_y:
            self.data_y = file['y']
            if to_one_hot:
                self.n_classes = self.data_y[:].max() + 1
        
        self.sampled_indexes = sampled_indexes
        if sampled_indexes is not None:
            self.num_examples = sampled_indexes.shape[0]
        else:
            self.num_examples = self.data_x.shape[0] # Number of examples in the dataset
        self.batch_size = batch_size
        self.steps_per_epoch = math.ceil(self.num_examples/self.batch_size)
        self.current_step = 0
        
        
           
    # Generator structure requires the __iter__ and a __next__ methods
    def __iter__(self):
        return self
    
    def __next__(self):
        #no stop condition! "fit_generator" assumes an endless generator
        
        step = self.current_step % self.steps_per_epoch
        
        if self.sampled_indexes is None:
            #sequential iteration over the data
            if (step+1) == self.steps_per_epoch:    #<- last epoch
                batch_indexes = list(range(step*self.batch_size, self.num_examples))
            else:
                batch_indexes = list(range(step*self.batch_size, (step+1)*self.batch_size))
        else:
            #sequential iteration over the sampled_indexes
            if (step+1) == self.steps_per_epoch:    #<- last epoch
                batch_indexes = sampled_indexes[step*self.batch_size : self.num_examples]
            else:
                batch_indexes = sampled_indexes[step*self.batch_size : (step+1)*self.batch_size]
            
        self.current_step += 1
        
        if self.hold_y:
            # Return the arrays in the shape that fit_gen uses (data, target)
            if self.to_one_hot:
                return (self.data_x[batch_indexes,...],
                        one_hot(self.data_y[batch_indexes,...], self.n_classes))
            else:
                return (self.data_x[batch_indexes,...],
                        self.data_y[batch_indexes,...])
        else:
            # Return the arrays in the shape that predict_generator uses (data)
            return (self.data_x[batch_indexes,...]) 
        # print("i shouldn't be here :D")
                
                
#for classification problems:                
def errors_h5(classifier, file_object, batch_size):

    x_generator = DataHolderH5(file_object, batch_size, None, hold_y = False)
    classification = classifier.predict_classes(x_generator)                        #<------------------------------------------------ parei aqui: o classifier é um model do keras, e não há generator para prever as classes
    c = numpy.argmax( one_hot(file_object['y'], file_object['y'].max()+1) , axis=1)        #TODO: check if this works
    r = numpy.where(classification != c, 1.0, 0.0)
    return r
    
def accuracy_h5(classifier, file_object, batch_size):
    e = errors_h5(classifier, file_object, batch_size)
    return 1.0 - (float(e.sum()) / float(file_object['y'].shape[0]))
 
#TODO: this is kinda a redefinition of data.py one_hot -> take care of the duplicates!
def one_hot(data, n_classes):
    b = np.zeros((data.size, n_classes),dtype='float32')
    b[np.arange(data.size), data] = 1.
    return b
    