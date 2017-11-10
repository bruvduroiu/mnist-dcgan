import struct
import os
import math

import keras
from keras.models import Sequential
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
    Reshape,
    UpSampling2D,
    LeakyReLU,
    Dropout,
)
from keras.optimizers import (
    Adam,
    SGD
)
from keras.utils import np_utils
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = (X_train.astype('float32') / 255, X_test.astype('float32') / 255)
X_train, X_test = X_train.reshape(X_train.shape[0], 1, 28, 28), X_test.reshape(X_test.shape[0], 1, 28, 28)
y_train, y_test = np_utils.to_categorical(y_train, 10), np_utils.to_categorical(y_test, 10)


def generator():
    model = Sequential()
    model.add(Dense(512*7*7, input_shape=(100,), kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape([512, 7, 7]))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform'))
    model.add(Activation('tanh'))
    return model
    

def discriminator():
    model = Sequential()
    model.add(Conv2D(256, (5, 5), input_shape=(28, 28, 1), strides=(2, 2), border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (5, 5), strides=(2, 2), border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


G = generator()
D = discriminator()
g_optim = Adam(lr=1e-5)
d_optim = SGD(lr=1e-2)

G.compile(loss='binary_crossentropy', optimizer=g_optim)
D.compile(loss='categorical_crossentropy', optimizer=g_optim)

print(G.summary())
print(D.summary())

GAN = Sequential()
GAN.add(G)
GAN.add(D)

GAN_optim = Adam(lr=1e-5)
GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optim)
print(GAN.summary())