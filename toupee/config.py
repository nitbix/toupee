#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

# import toupee.ensemble_methods
import yaml 
import toupee.parameters as parameters
import os

def yaml_include(loader, node):
    # Get the path out of the yaml file
    file_name = os.path.join(os.path.dirname(loader.name), node.value)

    with file(file_name) as inputfile:
        return yaml.load(inputfile)

yaml.add_constructor("!include", yaml_include)

defaults = { 'random_seed': None,
             'save_images': False,
             'join_train_and_valid': False,
             #TODO: 'pretraining_noise': None,
             'detailed_stats': False,
             'online_transform': None,
             'resize_data_to': None,
             'join_train_and_valid': False,
             'shuffle_dataset': False,
             'update_inputs': False,
             'update_inputs_lr': 1,
             #TODO:'pretraining': None,
             'early_stopping' : None,
             #TODO:'pretraining_passes' : 0,
             'one_hot' : False,
             'pickled' : False,
             'zca_whitening' : False,
             'test_at_each_epoch': True,
             'classification' : True,
           }

class Loader(yaml.Loader):

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
