import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
import pickle

np.random.seed(0)
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
batch_size = 64
num_epochs = 10

print("Load data...")
x_train = pickle.load(open('Trained Data/imdb/x_train.pkl', 'rb'))
y_train = pickle.load(open('Trained Data/imdb/y_train.pkl', 'rb'))
x_test = pickle.load(open('Trained Data/imdb/x_test.pkl', 'rb'))
y_test = pickle.load(open('Trained Data/imdb/y_test.pkl', 'rb'))
z = pickle.load(open('Trained Data/imdb/z.pkl', 'rb'))
model_input = pickle.load(open('Trained Data/imdb/inpu.pkl', 'rb'))

conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))