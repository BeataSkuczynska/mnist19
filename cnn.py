import os

import keras
from keras.layers import Dense, Flatten, Activation, LeakyReLU, Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from mlxtend.data import loadlocal_mnist
from skimage.transform import downscale_local_mean
import numpy as np
from time import time

from load_data import load_noisy_data


def reshape_data(data, image_downsize, height, width):
    new_data = np.zeros((data.shape[0], height * width))
    if image_downsize:
        for i, image in enumerate(data):
            new_image = image.reshape(28, 28)
            new_image = downscale_local_mean(new_image, (2, 2))
            new_image = new_image.reshape(1, height * width)
            new_data[i] = new_image
        new_data = new_data.reshape(new_data.shape[0], height, width, 1)
        return new_data
    else:
        data = data.reshape(data.shape[0], height, width, 1)
    return data


batch_size = 128
n_classes = 10
epochs = 10
activation = 'relu'  # 'relu' or 'leakyrelu'
optimizer = 'adadelta'  # 'adadelta' or 'sgd'
image_downsize = False

if image_downsize:
    # input image dimensions
    height, width = 14, 14
else:
    # input image dimensions
    height, width = 28, 28

# set the local directory in which the mnist data is saved
directory = '/home/komputerka/PycharmProjects/mnist19/samples'

# load the data in training and test set
data_train, label_train = loadlocal_mnist(images_path='{0}/train-images-idx3-ubyte'.format(directory),
                                          labels_path='{0}/train-labels-idx1-ubyte'.format(directory))
# data_train, label_train = load_noisy_data(os.path.join(directory, 'noisy'))

data_test, label_test = loadlocal_mnist(images_path='{0}/t10k-images-idx3-ubyte'.format(directory),
                                        labels_path='{0}/t10k-labels-idx1-ubyte'.format(directory))
# reshape the data
data_train = reshape_data(data_train, image_downsize, height, width)
data_test = reshape_data(data_test, image_downsize, height, width)
input_shape = (height, width, 1)

# the data is normalized between 0 and 1
data_train = data_train.astype('float32')
data_test = data_test.astype('float32')
data_train /= 255
data_test /= 255

# create class matrices for classification
label_train = keras.utils.to_categorical(label_train, n_classes)
label_test = keras.utils.to_categorical(label_test, n_classes)

# data augmentation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
datagen.fit(data_train)

train = datagen.flow(data_train, label_train, batch_size=batch_size)

# create model
model = keras.models.Sequential()

# first convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
if activation == 'relu':
    model.add(Activation('relu'))
elif activation == 'leakyrelu':
    model.add(LeakyReLU())

# first pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# second convolutional layer
model.add(Conv2D(64, (3, 3)))
if activation == 'relu':
    model.add(Activation('relu'))
elif activation == 'leakyrelu':
    model.add(LeakyReLU())

# second pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# reshape the activation to a 1-d vector
model.add(Flatten())

# first fully connected layer
model.add(Dense(128))
if activation == 'relu':
    model.add(Activation('relu'))
elif activation == 'leakyrelu':
    model.add(LeakyReLU())

# second fully connected layer (to the output)
model.add(Dense(n_classes, activation='softmax'))

# set optimizer
if optimizer == 'sgd':
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
elif optimizer == 'adadelta':
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

# time training:
start = time()

# training with augmented data
model.fit_generator(train, samples_per_epoch=len(data_train), epochs=epochs, verbose=2,
                    validation_data=(data_test, label_test))

# train model
# model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs, verbose=2,
# validation_data=(data_test, label_test))

elapsed = time() - start
print("Traing took {0} seconds".format(elapsed))

# test model
performance = model.evaluate(data_test, label_test, verbose=0)
print('Validation accuracy:', performance[1])
