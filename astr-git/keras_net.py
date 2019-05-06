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
mines_data,notmines_data,test,name_tags = load_dataset(dataset_name)
mines_data = increase_underrepresented(mines_data)

TEST = np.array(test)
X, Y = create_data_feed(mines_data,notmines_data)

print("Input images have shape: " +str(X.shape))
print("Test images have shape: " +str(TEST.shape))
X = X/255
TEST = TEST/255


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
# model = build_model(input_shape)
# model.fit(X, Y, batch_size=40, epochs=15, validation_split=0.2)
# model.save_weights('tf_models/last-attempt.tf')
# predictions = model.predict(TEST)
# input_shape = X.shape[1:]
def load_model_from_weights(tf_file,input_shape):
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
    model.load_weights(tf_file)
    predictions = model.predict(TEST)
    return predictions

def check_predicitons(list_predictions,name_tags):
    correct=0
    print("Prediction ----->")
    for i in range(len(list_predictions)):
        name = name_tags[i]
        mine = name.split('-')[0]
        if(list_predictions[i][1]>0.5 and mine=="Mine"):
            print("Mine found:")
            correct +=1
        elif(mine=="Mine"):
            print("Mine missed:")


        print(name_tags[i])
    print(correct)

tf1 = 'tf_models/last-attempt.tf'
tf2 = 'tf_models/third-attempt.tf'
tf3 = 'tf_models/last-attempt.tf'

pred = load_model_from_weights(tf1, input_shape)
check_predicitons(pred,name_tags)

pred = load_model_from_weights(tf2, input_shape)
check_predicitons(pred,name_tags)

pred = load_model_from_weights(tf3, input_shape)
check_predicitons(pred,name_tags)
