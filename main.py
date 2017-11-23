import struct
import os
import math
import numpy as np

import click

import keras
import tensorflow as tf
from keras.models import (
    Sequential,
    Model,
)
from keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
    Reshape,
    UpSampling2D,
    LeakyReLU,
    Dropout,
    ZeroPadding2D,
    Input,
)
from keras.optimizers import (
    Adam,
    SGD,
)
from keras.initializers import RandomNormal
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.datasets import mnist

from PIL import Image


class DCGAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.get_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.get_generator()
        self.generator.compile(loss='binary_crossentropy', 
                               optimizer=optimizer)

        noise = Input(shape=(100,))
        img = self.generator(noise)

        self.discriminator.trainable = False
        
        validity = self.discriminator(img)

        self.adversarial_pair = Model(noise, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)

    def get_generator(self):
        model = Sequential()
        model.add(Dense(128*7*7, input_shape=(100,)))
        model.add(Activation('relu'))
        model.add(Reshape([7, 7, 128]))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding='same'))
        model.add(Activation('tanh'))

        noise = Input(shape=(100,))
        img = model(noise)
        return Model(noise, img) 
    

    def get_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        img = Input(shape=(28, 28, 1))
        validity = model(img)
        return Model(img, validity)

    def save_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        generated_images = self.generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        fig, axes = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axes[i,j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
                axes[i,j].axis('off')
                cnt += 1
        fig.savefig("gen_%d.png" % epoch)

    def train(self, epochs, batch_size=128, save_interval=50):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.array([X_train[i] for i in range(len(X_train)) if y_train[i] == 0])
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train[:, :, :, None]
        X_test = X_test[:, :, :, None]

        half_batch = (batch_size / 2)

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0,1, (batch_size, 100))
            g_loss = self.adversarial_pair.train_on_batch(noise, np.ones((batch_size, 1)))

            print('{epoch} [D loss: {d_loss}, acc.: {d_accuracy}%]\t[G loss: {g_loss}'.format(epoch=epoch,
                                                                                              d_loss=d_loss[0],
                                                                                              d_accuracy=100*d_loss[1],
                                                                                              g_loss=g_loss))

            if epoch % save_interval == 0:
                self.save_images(epoch)

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=128, save_interval=50)