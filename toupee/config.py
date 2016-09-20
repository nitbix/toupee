#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import yaml 
import ensemble_methods
import parameters

defaults = { 'random_seed': None,
             'save_images': False,
             'pretraining_noise': None,
             'detailed_stats': False,
             'online_transform': None,
             'resize_data_to': None,
             'join_train_and_valid': False,
             'center_and_normalise': False,
             'shuffle_dataset': False,
             #TODO:'update_input': False,
             #TODO:'pretraining': None,
             'early_stopping' : None,
             'training_method' : 'normal',
             'pretraining_passes' : 0,
             'one_hot' : False,
           }

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
