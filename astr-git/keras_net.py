import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
####$$$$
import numpy as np
import random
import cv2
from utils.utils_hdf5 import load_dataset, store_h5
from data_augmentation import add_labels, increase_underrepresented

random.seed(777)

dataset_name "dataset.hdf5"
mines_data,notmines_data = load_dataset(dataset_name)
mines_data = increase_underrepresented(mines_data)
X, Y = add_labels(mines_data,notmines_data)


model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(13))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

    # Training the model, with 40 iterations
    # validation_split corresponds to the percentage of images used for the validation phase compared to all the images


# print(type(X[0]))
# X1 = X1
history = model.fit(X, Y, batch_size=32, epochs=40, validation_split=0.1)
