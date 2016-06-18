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
import learning_rates

defaults = { 'random_seed': None,
             'save_images': False,
             'pretraining_noise': None,
             'detailed_stats': False,
             'output_layer': 'softmax',
             'output_layer_options': {},
             'pretrain_update_rule': None,
             'pretrain_learning_rate': None,
             'online_transform': None,
             'resize_data_to': None,
             'join_train_and_valid': False,
             'center_and_normalise': False,
             'RGB': False,
             'shuffle_dataset': False,
             'update_input': False,
             'pretraining': None,
             'training_method' : 'normal',
             'pretraining_passes' : 0,
           }
def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
