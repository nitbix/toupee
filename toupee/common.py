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
                if 'serialize' in dir(o) and callable(getattr(o,'serialize')):
                    return o.serialize()
                if 'tolist' in dir(o) and callable(getattr(o,'tolist')):
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
    #euclidian_distance = sqrt{(y[0]-y_pred[0])^2 + (y[1]-y_pred[1])^2 + ... + (y[n-1]-y_pred[n-1])^2}
    prediction = predictor.predict(test_set_x)
    elementwise_d_squared = np.square(prediction - test_set_y)
    euclidian_distance = np.sqrt(np.sum(elementwise_d_squared, axis = 1))
    return euclidian_distance

def euclidian_distance(predictor, test_set_x, test_set_y):
    euclidian_distance = distance(predictor, test_set_x, test_set_y)
    return(np.sum(euclidian_distance) / float(test_set_y.shape[0]))
    
    
    

if 'toupee_global_instance' not in locals():
    toupee_global_instance = Toupee()
