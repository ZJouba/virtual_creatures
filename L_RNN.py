from tkinter import filedialog
import tkinter as tk
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

data = pickle.load(open(filepath, 'rb'))

data.drop_duplicates(subset='L-string', inplace=True)

data = data[['L-string', 'Area']]

data.reset_index(inplace=True)

X = data['Area'].values
Y = data['L-string']

""" --- Y data OneHotEncoding --- """
astr = list(Y.iloc[0])
to_dict = dict([(y, x+1) for x, y in enumerate(sorted(set(astr)))])
from_dict = {v: k for k, v in to_dict.items()}

new = Y.map(list)
frame = new.apply(pd.Series)
enc_Y = frame.applymap(to_dict.get)

# encoder = LabelEncoder()
# encoder.fit_transform(Y.reshape(-1, 1))  # .toarray()
# enc_Y = Y.apply(lambda x: ''.join([str(d.get(i)) for i in list(x)]))
# enc_Y = pd.to_numeric(enc_Y, errors='coerce')

""" --- X data StandardScaling --- """
scaler = StandardScaler()
enc_X = scaler.fit_transform(X.reshape(-1, 1))

index = int(enc_X.shape[0]*0.9)
X_train = enc_X[: index].flatten()
Y_train = enc_Y[: index]
X_test = enc_X[index:].flatten()
Y_test = enc_Y[index:]

""" --- KERAS ANN --- """
# model = Sequential()
# # model.add(LSTM(128, input_shape=(501, data.shape[0])))
# # model.add(Dense(data.shape[0]))
# # model.add(Activation('softmax'))
# model.add(Dense(501, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(501, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.fit(X_train, Y_train.values, batch_size=100, epochs=50)
# plot_model(model, to_file='model.png',
#            show_shapes=True, show_layer_names=True)
# img = Image.open('model.png')
# img.show()

""" --- LINEAR REGRESSION --- """
# model = LinearRegression()
# model.fit(X_train, Y_train)

""" --- SVM --- """
if not os.path.exists('model.p'):
    model = LinearSVR(C=1, loss='hinge', max_iter=1000, verbose=0)
    model.fit(X=Y_train, y=X_train)
    pickle.dump(model, open('model.p', 'wb'))
else:
    model = pickle.load(open('model.p', 'rb'))

Y_predict = model.predict(Y_test)
error = mean_squared_error(Y_predict, X_test.values)
print('\n {0:.0%}'.format(error))

lstring = np.array('F' * 501)
lstring = lstring.reshape(-1, 1)
# X_new = encoder.transform(lstring)
# max_area = scaler.transform(np.array((len(astr))).reshape(-1, 1))
max_area = np.linspace(min(enc_X), max(enc_X), 10).reshape(-1, 1)

max_predict = np.around(model.predict(max_area))

strings = []
pred_string = (np.vectorize(from_dict.__getitem__)(max_predict)).tolist()
for pred in pred_string:
    strings.append(''.join(pred))

areas = []
for area in max_area:
    areas.append(scaler.inverse_transform(area))

predictions = np.hstack(
    (np.asarray(areas), np.asarray(strings).reshape(-1, 1)))
pred_frame = pd.DataFrame(predictions, columns=[
                          'Input Area', 'Predicted L-string'])
pred_frame['Input Area'] = pd.to_numeric(pred_frame['Input Area'])
pred_frame['Approximate Area'] = pred_frame['Predicted L-string'].str.count(
    'F') + 0.7854
pred_frame['% Error'] = (abs(pred_frame['Input Area'] -
                             pred_frame['Approximate Area'])/pred_frame['Approximate Area'])*100
pred_frame.to_csv('predicted.csv')
