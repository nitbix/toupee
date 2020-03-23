#!/usr/bin/python

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D,\
    Convolution2D, MaxPooling2D, AveragePooling2D, Input, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

input_shape = [32,32,3]
output_classes = 10
l2_norm = 0.0001

input_layer = Input(shape=input_shape)

output = tf.keras.layers.ZeroPadding2D((1,1))(input_layer)
output = tf.keras.layers.Convolution2D(64, (3, 3), bias_regularizer=l2(l2_norm),
    kernel_regularizer=l2(l2_norm), padding='valid')(output)
output = tf.keras.layers.BatchNormalization(axis=1)(output)
output = tf.keras.layers.Activation('relu')(output)
output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)

output = tf.keras.layers.Convolution2D(128, (3, 3), bias_regularizer=l2(0.0001),
    kernel_regularizer=l2(0.0001), padding='valid')(output)
output = tf.keras.layers.BatchNormalization(axis=1)(output)
output = tf.keras.layers.Activation('relu')(output)
output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)

output = tf.keras.layers.Flatten()(output)
output = tf.keras.layers.Dense(1024)(output)
output = tf.keras.layers.BatchNormalization(axis=1)(output)
output = tf.keras.layers.Activation('relu')(output)
output = tf.keras.layers.Dropout(0.5)(output)

output = tf.keras.layers.Dense(output_classes)(output)
output = tf.keras.layers.BatchNormalization(axis=1)(output)
output = tf.keras.layers.Softmax()(output)

model = tf.keras.Model(inputs=input_layer, outputs=output)
print(model.to_yaml())
