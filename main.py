import struct
import os
import math
import numpy as np

import keras
import tensorflow as tf
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

from PIL import Image


def generator():
    model = Sequential()
    model.add(Dense(128*7*7, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape([7, 7, 128]))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    return model
    

def discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model


def adversarial_pair(G, D):
    model = Sequential()
    model.add(G)
    D.trainable = False
    model.add(D)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE=128):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    G = generator()
    D = discriminator()
    GAN = adversarial_pair(G, D)
    g_optim = Adam(lr=0.00002)
    d_optim = SGD(lr=0.00002, momentum=0.9, nesterov=True)
    G.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    GAN.compile(loss='binary_crossentropy', optimizer=g_optim, metrics=['accuracy'])
    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])

    print(G.summary())
    print(D.summary())
    print(GAN.summary())

    for epoch in range(10):
        print('Epoch is: ', epoch)
        print('Number of batches: ', int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = G.predict(noise, verbose=1)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5 
                Image.fromarray(image.astype(np.uint8)).save(
                    '{}_{}.png'.format(epoch, index))
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            D.trainable = True
            d_loss = D.train_on_batch(X, y)
            print('batch {} d_loss: {}'.format(index, d_loss))
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            D.trainable = False
            g_loss = GAN.train_on_batch(noise, [1] * BATCH_SIZE)
            D.trainable = True
            print('batch {} g_loss: {}'.format(index, g_loss))
            if index % 10 == 9:
                G.save_weights('generator.h5', True) 
                D.save_weights('discriminator.h5', True) 


train()
