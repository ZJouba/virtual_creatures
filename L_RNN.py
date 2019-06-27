from tkinter import filedialog
import tkinter as tk
import sklearn
import pickle
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

data = pickle.load(open(filepath, 'rb'))

data.drop_duplicates(subset='Area', inplace=True)

data = data[['L-strings', 'Area']]

features = data['Area']
targets = data['L-strings']

encoder = OneHotEncoder()
enc_targets = encoder.fit_transform(targets)

model = Sequential()
model.add(LSTM(128, input_shape=(501, data.shape[0])))
model.add(Dense(data.shape[0]))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(features, targets, batch_size=128, epochs=5)

max_predict = model.predict(500)
max_predict_lstring = encoder.inverse_transform(max_predict)
