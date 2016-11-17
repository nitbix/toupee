#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import copy

to_string = ['method']
class Parameters(object):

    def __init__(self, **entries): 
        self.__dict__.update(entries)

    def serialize(self):
        serialized = copy.deepcopy(self)

        for x in to_string:
            if x in serialized.__dict__:
                serialized.__dict__[x] = serialized.__dict__[x].serialize()
        return serialized.__dict__
