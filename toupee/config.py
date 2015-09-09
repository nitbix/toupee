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

def load_parameters(filename):
    with open(filename) as f:
        r = yaml.load(f)
    if 'random_seed' not in r:
        r['random_seed'] = [2013,1,4]
    return parameters.Parameters(**r)
