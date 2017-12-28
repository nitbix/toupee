"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy as np
import os

def gauss(x, y, sigma=2.0):
    Z = 2 * np.pi * sigma**2
    return  1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))
