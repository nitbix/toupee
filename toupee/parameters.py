#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import copy

to_string = ['cost_function', 'learning_rate', 'update_rule']
class Parameters(object):

    def __init__(self, **entries): 
        self.__dict__.update(entries)

    def serialize(self):
        serialized = copy.deepcopy(self)
        for layer in serialized.n_hidden:
            s_layer = [str(param) for param in layer[1]]
            layer[1] = s_layer
        for x in to_string:
            serialized.__dict__[x] = serialized.__dict__[x].serialize()
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        return serialized.__dict__
                
