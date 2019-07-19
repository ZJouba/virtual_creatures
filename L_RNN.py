import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend
from tkinter import filedialog
import tkinter as tk
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, LinearSVR
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, ELU, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
# from keras_tqdm import TQDMCallback
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import time

root = tk.Tk()
root.attributes("-topmost", True)
root.withdraw()
filepath = filedialog.askopenfilename()

print('\nFile selected. Preprocessing')
big_data = pickle.load(open(filepath, 'rb'))

# big_data.sort_values(by=['Area'], ascending=False, inplace=True)

# max_area = big_data['Area'].head(1).values

data = pd.DataFrame(big_data['Rules'].apply(pd.Series).values.ravel())
data.columns = ['Rule']
data['Area'] = big_data['Ratios'].apply(pd.Series).values.ravel()

# data.drop_duplicates(subset=['Rule'], inplace=True)

# data.reset_index(inplace=True, drop=True)

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

""" --- LINEAR REGRESSION --- """
# model = LinearRegression(n_jobs=-1)

""" --- SVM --- """
# model = SVR(kernel="poly", gamma='scale', degree=1, C=100, epsilon=0.1)

# if not os.path.exists('model.p'):
#     model = LinearSVR(C=1, loss='hinge', max_iter=1000, verbose=0)
#     model.fit(X=X_train, y=Y_train)
#     pickle.dump(model, open('model.p', 'wb'))
# else:
#     model = pickle.load(open('model.p', 'rb'))

print('\nTraining')
# model.fit(X_train, Y_train)

""" --- KERAS ANN --- """
def evaluate_model(layers, X_train, Y_train, X_test, Y_test):

    # for acti in ann_params.get('activation'):
    # model_path = os.path.join(os.curdir, 'Keras_ANN.p')

    # mc = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
    es = EarlyStopping(monitor='loss', verbose=1, patience=5)

    nodes = 1000 # enc_Y_1.shape[0]
    features = Y_train.shape[1]

    model = Sequential()

    model.add(Dense(nodes, input_shape=(1,), activation='elu'))

    for _ in range(1, layers):
        model.add(Dense(nodes, activation='elu'))

    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    # model.add(Dense(top_nodes//3))
    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    # model.add(Dense(top_nodes//6))
    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    # model.add(Dense(10))
    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    # model.add(Dense(top_nodes//6))
    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    # model.add(Dense(top_nodes//3))
    # model.add(BatchNormalization())
    # model.add(ELU())
    # model.add(Dropout(rate=0.2))

    model.add(Dense(features, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=100, epochs=100, callbacks=[es], verbose=0)

    _, test_acc = model.evaluate(X_test, Y_test, verbose=1)

    return history, test_acc

# layers = [16,18,20]

# for ind, layer in enumerate(layers):
#     history, result = evaluate_model(layer, X_train, Y_train, X_test, Y_test)

#     print('#%d: layers = %d: %.3f' % (ind, layer, result))

#     plt.plot(history.history['loss'], label=str(layer))

# plt.xlabel('Epoch')
# plt.ylabel('Cross entropy loss')
# plt.legend()
# plt.show()

es = EarlyStopping(monitor='loss', verbose=1, patience=5)

nodes = 1000
features = Y_train.shape[1]

model = Sequential()

model.add(Dense(500, input_shape=(1,)))
model.add(BatchNormalization())
model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.25)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.5)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.75)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//2)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//2.5)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//3)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//4)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(10))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//4)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//3)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//2.5)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//2)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.75)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.5)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

# model.add(Dense(int(nodes//1.25)))
# model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dropout(rate=0.2))

model.add(Dense(features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=1000, epochs=500, callbacks=[es], verbose=1)

_, test_acc = model.evaluate(X_test, Y_test, verbose=1)

print(test_acc)

""" --- KERAS RNN --- """

Y_predict = model.predict(X_test)
error = mean_squared_error(Y_predict, Y_test)
print('\nMSE: {0:.3}'.format(error))

areas = np.arange(0, 600, 100)
for area in areas:
    pred_area = scaler.transform(np.array(area).reshape(1,-1))
    prediction = model.predict(pred_area)
    string = binarizer.inverse_transform(prediction)

    print('Area: {} \t String: {}'.format(area, string))

# pred_string = ''.join([from_dict.get(c, ' ') for c in max_predict.tolist()[0]])

# print(string)