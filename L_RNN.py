from tkinter import filedialog
import tkinter as tk
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, LinearSVR
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

print('\nPreprocessing')
big_data = pickle.load(open(filepath, 'rb'))

big_data.sort_values(by=['Area'], ascending=False, inplace=True)

max_area = big_data['Area'].head(1).values

data = pd.DataFrame(big_data['Rules'].apply(pd.Series).values.ravel())
data.columns = ['Rule']
data['Area'] = big_data['Ratios'].apply(pd.Series).values.ravel()

data.drop_duplicates(subset=['Rule'], inplace=True)

data.reset_index(inplace=True, drop=True)

X = data['Area'].values
Y = data['Rule']

""" --- Y data OneHotEncoding --- """
chars = [
    'F',
    '+',
    '-',
    'X',
]
to_dict = dict([(y, x+1) for x, y in enumerate(sorted(set(chars)))])
from_dict = {v: k for k, v in to_dict.items()}

new = Y.map(list)
enc_Y = new.apply(lambda x: np.array(list(filter(None, map(to_dict.get, x)))))
enc_Y = np.stack(enc_Y, axis=0)

binarizer = LabelBinarizer()
enc_Y_1 = binarizer.fit_transform(Y.values.reshape(-1,1))

""" --- X data StandardScaling --- """
scaler = StandardScaler()
enc_X = scaler.fit_transform(X.reshape(-1, 1))

index = int(enc_X.shape[0]*0.9)
X_train = enc_X[: index]
Y_train = enc_Y_1[: index]
X_test = enc_X[index:]
Y_test = enc_Y_1[index:]

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
# model = LinearRegression(n_jobs=-1)

""" --- SVM --- """
model = SVR(kernel="poly", gamma='scale', degree=1, C=100, epsilon=0.1)

# if not os.path.exists('model.p'):
#     model = LinearSVR(C=1, loss='hinge', max_iter=1000, verbose=0)
#     model.fit(X=X_train, y=Y_train)
#     pickle.dump(model, open('model.p', 'wb'))
# else:
#     model = pickle.load(open('model.p', 'rb'))

print('\nTraining')
model.fit(y=X_train, X=Y_train)

test = binarizer.transform(['FFFFX'])
Y_predict = model.predict(test)
# error = mean_squared_error(Y_predict, Y_test)
# print('\nMSE: {0:.3%}'.format(error))
print(scaler.inverse_transform(Y_predict))

test = binarizer.transform(['++++X'])
Y_predict = model.predict(test)
print(scaler.inverse_transform(Y_predict))

test = binarizer.transform(['FFFFF'])
Y_predict = model.predict(test)
print(scaler.inverse_transform(Y_predict))

# areas = [0, 50, 100, 150, 200, 250]
# for area in areas:
#     pred_area = scaler.transform(np.array(area).reshape(1,-1))
#     prediction = model.predict(pred_area)
#     string = binarizer.inverse_transform(prediction)

#     print('Area: {} \t String: {}'.format(area, string))

# pred_string = ''.join([from_dict.get(c, ' ') for c in max_predict.tolist()[0]])

# print(string)