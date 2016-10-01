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

input_shape = [3,32,32]
output_classes = 10

model = Sequential()

#Model.add(Convolution2D(32, 3, 3, border_mode='same',
#                            input_shape=input_shape))
#Model.add(Activation('relu'))
#Model.add(Convolution2D(32, 3, 3))
#Model.add(Activation('relu'))
#Model.add(MaxPooling2D(pool_size=(2, 2)))
#Model.add(Dropout(0.25))
#
#Model.add(Convolution2D(64, 3, 3, border_mode='same'))
#Model.add(Activation('relu'))
#Model.add(Convolution2D(64, 3, 3))
#Model.add(Activation('relu'))
#Model.add(MaxPooling2D(pool_size=(2, 2)))
#Model.add(Dropout(0.25))
#
#Model.add(Flatten())
#Model.add(Dense(512))
#Model.add(Activation('relu'))
#Model.add(Dropout(0.5))
#Model.add(Dense(output_classes))
#Model.add(Activation('softmax'))
#
#all -cnn
#TODO: add L2 reg
model.add(ZeroPadding2D((1,1), input_shape=input_shape))
model.add(Convolution2D(96, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(96, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(96, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), subsample=(2,2)))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(192, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(192, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(192, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), subsample=(2,2)))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(192, 3, 3, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(192, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(10, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))


model.add(AveragePooling2D((8,8)))
model.add(Flatten())

#model.add(Dense(output_classes))
#model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('softmax'))

print model.to_yaml()
