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
             'join_train_and_valid': False,
             #TODO: 'pretraining_noise': None,
             'detailed_stats': False,
             'online_transform': None,
             'resize_data_to': None,
             'join_train_and_valid': False,
             'shuffle_dataset': False,
             #TODO:'update_input': False,
             #TODO:'pretraining': None,
             'early_stopping' : None,
             #TODO:'pretraining_passes' : 0,
             'one_hot' : False,
             'pickled' : False,
             'zca_whitening' : False,
           }

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
