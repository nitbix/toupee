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

defaults = { 'random_seed': [2013,1,4],
             'save_images': False,
             'pretraining_noise': None,
             'detailed_stats': False
           }
def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    for d in defaults:
        if d not in r:
            r[d] = defaults[d]
    return parameters.Parameters(**r)
