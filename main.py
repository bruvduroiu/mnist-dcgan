import os

# Linear algebra and plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpltoolkits.mplot3d import Axes3D
from scipy.stats import norm

import tensorflow as tf

import seaborn as sns

# Deep Learning library
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# Import DCGAN
from models.mnist_dcgan import DCGAN

# Parameters

img_rows, img_cols, img_chns = 28, 28, 1
filters = 64
n_conv = 3
batch_size = 100
latent_dim = 3
intermediate_dim = 128
epochs = 5
epsilon_std = 1.0

WEIGHTS_FILE = 'weights/mnist_vae.h5'
GEN_WEIGHTS = 'weights/mnist_dcgan_generator.h5'
DIS_WEIGHTS = 'weights/mnist_dcgan_discriminator.h5'

original_img_size = (img_rows, img_cols, img_chns)


# Build the Network

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns, 
                kernel_size=(2, 2),
                padding='same',
                activation='relu')(x)
conv_2 = Conv2D(filters, 
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters, 
                kernel_size=n_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters, 
                kernel_size=n_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])


decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')

output_shape = (batch_size, 14, 14, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=n_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=n_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')

output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3 = Conv2DTranspose(filters,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding='valid',
                                   activation='relu')
decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

vae = Model(x, x_decoded_mean_squash)

xent_loss = img_rows * img_cols * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean_squash))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(loss=[None], optimizer='rmsprop')
vae.summary()

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255

X_train = X_train[:,:,:,None]
X_test  = X_test[:,:,:,None]

# Train the model

with tf.device('/gpu:0'):
    if os.path.exists(WEIGHTS_FILE):
        vae.load_weights(WEIGHTS_FILE)
    else:
        vae.fit(X_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, None))
        vae.save(WEIGHTS_FILE)


    dcgan = DCGAN()
    if os.path.exists(GEN_WEIGHTS) and os.path.exists(DIS_WEIGHTS):
        dcgan.load_weights(DIS_WEIGHTS, GEN_WEIGHTS)
    else:
        dcgan.train(epochs=4000, batch_size=128, save_interval=50)
        dcgan.save_weights(DIS_WEIGHTS, GEN_WEIGHTS)

    encoder = Model(x, z_mean)

    x_test_encoded = encoder.predict(X_test, batch_size=batch_size)

    x_gen_imgs = dcgan.generate_sample(10000)
    x_encoded_imgs = encoder.predict(x_gen_imgs, batch_size=10000)

    def get_gaussian(digit):
        idx = [i for i in range(y_test.shape[0]) if y_test[i] == digit]
        encoded_test = x_test_encoded[idx]
        
        mu = np.mean(encoded_test, axis=0)
        sigma = np.cov(encoded_test.T)
        
        return lambda x : (1 / (2*np.pi) * np.sqrt(np.linalg.det(sigma))) * np.exp( -0.5 * (x - mu).dot(np.linalg.inv(sigma)).dot((x - mu).T))

    DIGITS = [i for i in range(10)]
    GAUSSIANS = dict([[i, get_gaussian(i)] for i in DIGITS])
    PRIORS = {}
    total = y_test.shape[0]
    PRIORS = dict([[i, sum([1 for x in y_test if x == i])/total] for i in DIGITS])

    def classify_with_confidence(latent_vectors):

        classifications = []
        for x in latent_vectors:
            max_label = -1
            max_classif = -1
            for label, gaussian in GAUSSIANS.items():
                classif = (PRIORS[label] * gaussian(x)) / sum([PRIORS[i] * GAUSSIANS[i](x) for i in DIGITS])
                if classif > max_classif:
                    max_label = label
                    max_classif = classif
            print('Classification: {}; confidence: {}'.format(max_label, max_classif))
            classifications.append(max_label)
        return classifications

    def save_imgs_with_labels(images, labels):
        r, c = 5, 5

        fig, axes = plt.subplots(r, c, figsize=(20,20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                cnt = np.random.randint(0, images.shape[0])
                axes[i,j].imshow(images[cnt, :, :, 0], cmap='gray')
                axes[i,j].axis('off')
                axes[i,j].set_title(labels[cnt])
        fig.savefig("images_sample.png")

    def save_latent_space_distribution(encoded, encoded_labels, generated, generated_labels):
        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')

        ax.scatter(encoded[:,0], encoded[:,1], encoded[:,2], c=encoded_labels, cmap='jet')
        ax.scatter(generated[:,0], generated[:,1], generated[:,2],
                   s=400,
                   c=generated_labels,
                   marker='H',
                   linewidths=2,
                   edgecolors='y')

        # for i, txt in enumerate(generated_labels):
        #     ax.annotate(txt, xy=(generated[i,0], generated[i,1]), xytext=(generated[i,0]+0.5, generated[i,1]+0.5))

        fig.savefig('latent_space.png')

    classified_imgs = classify_with_confidence(x_encoded_imgs)
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(classified_imgs, color='b')
    fig.savefig('hist_gen.png')

    save_imgs_with_labels(x_gen_imgs, classified_imgs)

    save_latent_space_distribution(x_test_encoded, y_test, x_encoded_imgs[:10], classified_imgs[:10])
