'''
Code written by:
Dr. Avishai Sintov
Robotics lab, Tel-Aviv University
Email: sintov1@tauex.tau.ac.il
May 2023
'''

from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pickle
import numpy as np
import tensorflow as tf
import time

from grasps_3d import grasps
from shuffle_unison import shuffle_in_unison

H = [307] * 8
num_fingers = 4
with_normals = '_withN'

checkpoint_path = './tf_models/model_' + str(num_fingers) + with_normals + '.ckpt'

# load the dataset
with open('./data/id_data_' + str(num_fingers) + with_normals + '.pkl', 'rb') as H:
    X, y, objs = pickle.load(H)
X, y = shuffle_in_unison(X, y)

# ensure all data are floating point values
X = X.astype('float32')

# encode strings to integer
y = LabelEncoder().fit_transform(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]


print('Using model of size: ', H)

# define model
model = Sequential()
model.add(Dense(H[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
for j in range(1,len(H)):
    model.add(Dense(H[j], activation='relu', kernel_initializer='he_normal'))
    # model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(len(objs), activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
if 1:
    model.load_weights(checkpoint_path)
else:
    # model.load_weights(checkpoint_path)
    model.fit(X_train, y_train, epochs=40, batch_size=200, verbose=1)
    model.save_weights(checkpoint_path)

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc, H)


