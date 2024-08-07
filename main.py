import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "ISLDetection-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Loading Images
data = []
categories = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
i = 1
j = 1
input_dir = "E:/DATA/ISL 2.0/data1/train"

for category in categories:
    path = os.path.join(input_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      new_array = cv2.resize(img_array, (60, 60))
      data.append([new_array,class_num])



plot_class_distribution(input_dir)


import random

random.shuffle(data)


import random

random.shuffle(data)

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 60, 60, 1)
X = X/255 # Normalize Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categories = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(categories))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(categories))

# Building the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(categories)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=13, batch_size=32, validation_split=0.2, callbacks = [tensorboard])

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy: ", test_acc)

# Saving the model
model.save("sign_language_model.h5")
#Load Image
import io
from PIL import Image
import numpy as np

image_path = "E:/DATA/ISL 2.0/data1/test/1/0.jpg"
# Read and Preprocess Image
image = Image.open(image_path).convert('L')
image = image.resize((60, 60))
image_array = np.array(image)
image_array = image_array.reshape((-1, 60, 60, 1)).astype('float32') / 255

# Predict
predictions = model.predict(image_array)
predicted_sign = np.argmax(predictions)
print("Predicted Sign : ")
if predicted_sign == 0:
    print("0")
elif predicted_sign == 1:
    print("1")
elif predicted_sign == 2:
    print("2")
elif predicted_sign == 3:
    print("3")
elif predicted_sign == 4:
    print("4")
elif predicted_sign == 5:
    print("5")
elif predicted_sign == 6:
    print("6")
elif predicted_sign == 7:
    print("7")
elif predicted_sign == 8:
    print("8")
elif predicted_sign == 9:
    print("9")
elif predicted_sign == 10:
    print("A")
elif predicted_sign == 11:
    print("B")
elif predicted_sign == 12:
    print("C")
elif predicted_sign == 13:
    print("D")
elif predicted_sign == 14:
    print("E")
elif predicted_sign == 15:
    print("F")
elif predicted_sign == 16:
    print("G")
elif predicted_sign == 17:
    print("H")
elif predicted_sign == 18:
    print("I")
elif predicted_sign == 19:
    print("J")
elif predicted_sign == 20:
    print("K")
elif predicted_sign == 21:
    print("L")
elif predicted_sign == 22:
    print("M")
elif predicted_sign == 23:
    print("N")
elif predicted_sign == 24:
    print("O")
elif predicted_sign == 25:
    print("P")
elif predicted_sign == 26:
    print("Q")
elif predicted_sign == 27:
    print("R")
elif predicted_sign == 28:
    print("S")
elif predicted_sign == 29:
    print("T")
elif predicted_sign == 30:
    print("U")
elif predicted_sign == 31:
    print("V")
elif predicted_sign == 32:
    print("W")
elif predicted_sign == 33:
    print("X")
elif predicted_sign == 34:
    print("Y")
elif predicted_sign == 35:
    print("Z")
else:
    print("Unpredictable")

