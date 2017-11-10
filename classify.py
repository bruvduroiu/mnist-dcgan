import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Conv2D,
)
from keras.optimizers import (
    SGD,
    Adam,
    RMSProp,
)
from keras.datasets import mnist
from keras.utils import np_utils


batch_size = 128
nr_classes = 10
nr_iterations = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, nr_classes)
y_test = np_utils.to_categorical(y_test, nr_classes)

def classifier():
    model = Sequential()
    model.add()