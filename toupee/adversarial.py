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


def FGSM(model: toupee.model.Model, X: np.array, Y: np.array) -> np.array:
    with tf.GradientTape() as tape:
        X = tf.cast(X, tf.float32)
        tape.watch(X)
        prediction = model._model(X)
        loss = model._loss(Y, prediction)
        gradient = tape.gradient(loss, X)
        signed_grad = tf.sign(gradient)
        return signed_grad.numpy()