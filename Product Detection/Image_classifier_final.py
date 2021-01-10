# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:10:11 2020

@author: user
"""

import numpy as np
import pandas as pd
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras


file_list = []
class_list = []

DATADIR = "C:/Users/jiakj/Desktop/NUS/External Python and Co/Shopee/Competition/Shopee 20 June Product Detectionn Competition/Copy_of_shopee_product_detection_dataset/train/train"
#C:/Users/jiakj/Desktop/NUS/External Python and Co/Shopee/Competition/Shopee 20 June Product Detectionn Competition/Copy_of_shopee_product_detection_dataset/train/train

# All the categories you want your neural network to detect
CATEGORIES = ["00", "01", "02", "03", "04",
	      "05", "06", "07", "08", "09",
	      "10", "11", "12", "13", "14", "15", "16",
          "17", "18", "19", "20", "21", "22", "23",
          "24", "25", "26", "27", "28", "29", "30",
          "31", "32", "33", "34", "35", "36", "37",
          "38", "39", "40", "41"]

# The size of the images that your neural network will use
IMG_SIZE = 28

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
X = X / 255.0
y = np.array(y) #first edit

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)), #flatten to 1-dimension list, input we feeding, 28 x 28 pixels
    keras.layers.Dense(128, activation='relu'), #128 hidden units
    keras.layers.Dense(42) #output units
])

model.compile(optimizer='adam', #optimization algorithm
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #Goal is to minimize this loss function during training
              metrics=['accuracy'])

model.fit(X, y, epochs=10)

TESTDIR = "C:/Users/jiakj/Desktop/NUS/External Python and Co/Shopee/Competition/Shopee 20 June Product Detectionn Competition/Copy_of_shopee_product_detection_dataset/train/train"

test_data = []
filenames = []
class_predictions = []
test_data_in_csv = pd.read_csv("C:/Users/jiakj/Desktop/NUS/External Python and Co/Shopee/Competition/Shopee 20 June Product Detectionn Competition/Copy_of_shopee_product_detection_dataset/test.csv")
test_data_filenames = test_data_in_csv['filename'].values.tolist()



def create_test_data():
    class_num = 0
    for img in os.listdir(TESTDIR):
        try:
            if img in test_data_filenames:
                filenames.append(img)
                img_array = cv2.imread(os.path.join(TESTDIR, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append(new_array)
        except Exception as e:
            pass

create_test_data()

X_test = [] #features
#y_test = [0]*len(filenames) #labels
y_test = [0 for ele in range(len(filenames))]

for features in test_data:
    X_test.append(features)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE)
X_test = X_test / 255.0

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
#softmax layer normalizes output into a probability distribution

predictions = probability_model.predict(X_test)
for i in range(0, len(predictions)):
    class_num = np.argmax(predictions[i])
    y_test[i] = class_num
    class_predictions.append(class_num)

filename_df = pd.DataFrame(filenames, columns=['filename'])
category_df = pd.DataFrame(class_predictions, columns=['category'])
full_df = pd.concat([filename_df, category_df], axis=1)
full_df.to_csv('test_predictions.csv', index = False)
#test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

#print('\nTest accuracy:', test_acc)