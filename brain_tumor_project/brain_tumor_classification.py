import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import csv

input_base_path = r"C:\Users\Selim\Downloads\brain_tumor\Training"
classes = ["pituitary_tumor","no_tumor","meningioma_tumor","glioma_tumor"]
X = []
Y = []
image_width = 224
image_height = 224

os.chdir(input_base_path)
for i in classes:
    os.chdir(i)
    for files in os.listdir("./"):
        img = cv2.imread(files) #resimleri binary formata cevirdik
        img = cv2.resize(img, (image_height, image_width))
        X.append(img)
        Y.append(i)
    os.chdir("..")

X = np.array(X)
Y = np.array(Y)






datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
datagen.fit(X)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)
X = X.astype("float32")/255


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(image_height,image_width,3),activation="relu"))
model.add(Conv2D(64, (3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3),activation="relu"))
model.add(Conv2D(64, (3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(datagen.flow(X, Y, batch_size=64),epochs=50) #81.30 accuracy oranı

































































