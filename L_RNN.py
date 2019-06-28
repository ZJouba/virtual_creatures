import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.layers import LSTM, RNN
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
import pickle
import sklearn
import tkinter as tk
from tkinter import filedialog
import numpy as np


root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

data = pickle.load(open(filepath, 'rb'))

data.drop_duplicates(subset='L-string', inplace=True)

data = data[['L-string', 'Area']]

features = data['Area'].values
targets = data['L-string'].values

encoder = OneHotEncoder(categories='auto', sparse=False)
enc_targets = encoder.fit_transform(targets.reshape(-1,1)).reshape(2082, 2082, 1)

model = Sequential()
# model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(data.shape[0], input_shape=(2082, 2082, 1), activation='softmax'))
model.add(Dense(501, activation='softmax'))
model.add(Dense(3, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(features, targets, batch_size=128, epochs=5, verbose=1)

max_predict = model.predict(500)
max_predict_lstring = encoder.inverse_transform(max_predict)

print(max_predict_lstring)
