#!/usr/bin/python

import keras

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2

import sys

input_shape = [601]
output_classes = 4

#############
# Model settings
# hidden_layers = 1
# neurons_hidden = 1024

hidden_layers = int(sys.argv[1])    # <------ parameter being tested
neurons_hidden = int(sys.argv[2])   # <------ parameter being tested
#############

model = Sequential()

for i in range(hidden_layers):
    model.add(Dense(neurons_hidden,
        input_shape = input_shape,
        bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001),
        kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))



model.add(Dense(output_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# print model.to_yaml()
with open('dnn.model', 'w') as f:
    sys.stdout = f
    print(model.to_yaml())
