import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

root_dir = "dataset/images/images"

files = os.path.join(root_dir)
File_names = os.listdir(files)

## Run the below cells as it is
data = pd.read_csv("dataset/pokemon.csv")

## Run the below cells as it is
data_dict = {}
for key, val in zip(data["Name"], data["Type1"]):
    data_dict[key] = val

labels = data["Type1"].unique()

ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
labels_idx = dict(zip(labels, ids))

final_images = []
final_labels = []
count = 0
files = os.path.join(root_dir)
for file in File_names:
    count += 1
    img = cv2.imread(os.path.join(root_dir, file), cv2.COLOR_BGR2GRAY)
    label = labels_idx[data_dict[file.split(".")[0]]]
    # append img in final_images list
    final_images.append(np.array(img))
    # append label in final_labels list
    final_labels.append(np.array(label))

# converting lists into numpy arrayn
# normalizing and reshaping the data
final_images = np.array(final_images, dtype=np.float32) / 255.0
final_labels = np.array(final_labels, dtype=np.int8).reshape(809, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(120, 120, 3)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(18)
])
# print model summary and check trainable parameters
model.summary()

# compile model (Use: Adam optimizer, categorical_crossentropy loss and metrics as Accuracy)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit model (use images and labels)
history = model.fit(final_images, final_labels, epochs=50)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(final_images)


def predict_output(img_path):
    img_tmp = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    chosen_img = np.array(img_tmp)
    chosen_img_labels = np.array(label)
    final_images = np.array([chosen_img], dtype=np.float32) / 255.0
    print(probability_model.predict(final_images))



predict_output("dataset/images/images/axew.png")