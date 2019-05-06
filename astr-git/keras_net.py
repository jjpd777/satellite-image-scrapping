import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
# ####$$$$
import numpy as np
import random
import cv2
from pathlib import Path
from utils.utils_hdf5 import load_dataset
from utils.image_utils import create_data_feed, increase_underrepresented, feature_format


dataset_name ="dataset.hdf5"
mines_data,notmines_data,test,nms = load_dataset(dataset_name)
mines_data = increase_underrepresented(mines_data)

TEST = np.array(test)
X, Y = create_data_feed(mines_data,notmines_data)

print("Input images have shape: " +str(X.shape))
print("Test images have shape: " +str(TEST.shape))



# X = feature_format(X)
# TEST = feature_format(TEST)

print("Input images have shape: " +str(X.shape))
print("Test images have shape: " +str(TEST.shape))
X = X/255
TEST = TEST/255
#
def build_model(input_shape):
    layers = [tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=input_shape),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)]

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model
# #
# #
input_shape = X.shape[1:]
model = build_model(input_shape)
model.fit(X, Y, batch_size=60, epochs=10, validation_split=0.2)
model.save_weights('tf_models/third-attempt.tf')
predictions = model.predict(TEST)
# input_shape = X.shape[1:]
# #
# layers = [tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=input_shape),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
#     tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)]
# model = tf.keras.Sequential(layers)
# model.load_weights("midnight.tf")
#
# predictions = model.predict(TEST)
correct=0

for i in range(len(predictions)):
    # print(nms[i])
    # print(predictions[i])

    if(predictions[i][1]>0.5):
        print("Mine found:")

        correct +=1
        print(nms[i])
print(correct)


# # test_path = "dataset/clean/test/"
# # test_set = Path(test_path)
# # list_test = test_set.glob('*.jpg')
# # rest = []
# # for test in list_test:
# #     im = cv2.imread(str(test))/255
# #     rest.append(im)
# # rest = feature_format(rest)
#
# model.predict(rest)
#
# # im = np.expand_dims(im,axis=-1)

    # print(im.shape)

    # im = np.expand_dims(im,axis=-1)
    # img = im[0:580,200:1180].copy()
    # img = feature_format([img])
    # prediction = model.predict(im)
    # # name = str(test).split('/')[-1]
    #
    # if(np.argmax(prediction[0])==1):
    #     print("Mine")
    #     print(name)

    # print("Mine" if np.argmax(prediction[0])==1 else "Not Mine")
# path_to_hdf5 = "weights_output.hdf5"
# model.save_weights(path_to_hdf5)
# build_model(X,Y)
