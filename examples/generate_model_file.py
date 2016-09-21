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

input_shape = [3,32,32]
output_classes = 10

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_classes))
model.add(Activation('softmax'))

#all -cnn
#model.add(ZeroPadding2D((1,1), input_shape=input_shape))
#model.add(Convolution2D(96, 3, 3, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(96, 3, 3, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(96, 3, 3, subsample=(2,2)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(192, 3, 3, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(192, 3, 3, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(192, 3, 3, subsample=(2,2)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(192, 3, 3, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Convolution2D(192, 1, 1, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Convolution2D(10, 1, 1, border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#
#model.add(Flatten())
##model.add(Dense(1024))
##model.add(Activation('softmax'))
#model.add(Dense(output_classes))
#model.add(Activation('softmax'))
#
##model.add(Flatten())
##model.add(AveragePooling2D((6,6)))
#model.add(Activation('softmax'))

print model.to_yaml()
