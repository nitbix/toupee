#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import yaml 
import activations
import update_rules
import ensemble_methods
import parameters
import cost_functions
import activations
import weight_inits

defaults = { 'random_seed': None,
             'save_images': False,
             'pretraining_noise': None,
             'detailed_stats': False,
             'output_layer': 'softmax',
             'pretrain_update_rule': None,
             'pretrain_learning_rate': None,
             'online_transform': None,
             'resize_data_to': None,
             'subtract_mean': False,
             'join_train_and_valid': False,
             'RGB': False,
           }
def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
