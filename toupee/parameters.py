#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

class Parameters(object):

    def __init__(self, **entries): 
        self.__dict__.update(entries)
