import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

# get the data
X = pickle.load(open("./data/X.pickle", "rb"))
y = pickle.load(open("./data/y.pickle", "rb"))

X = X / 255.0

# build model
# layer 1
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu")) #ReLU is the most commonly used activation function in neural networks, especially in CNNs. If you are unsure what activation function to use in your network, ReLU is usually a good first choice.
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 3
model.add(Flatten())
model.add(Dense(64))

# ouput layer
model.add(Dense(1))
model.add(Activation("sigmoid"))#In artificial neural networks, sometimes non-smooth functions are used instead for efficiency; these are known as hard sigmoids.

# compile
model.compile(loss="binary_crossentropy",
              optimizer="adam", #Adadelta. adagradDA, adagrad?
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)