#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import toupee


def FGSM(model: toupee.model.Model, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # WARNING: this treats everything as a single batch!
    with tf.GradientTape() as tape:
        X = tf.convert_to_tensor(X)
        tape.watch(X)
        prediction = model._model(X)
        loss = model._loss(Y, prediction)
        gradient = tape.gradient(loss, X)
        signed_grad = tf.sign(gradient)
    return signed_grad.numpy()