from types import SimpleNamespace

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

import sys
import random

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from Tools.Classes import Creature

def create_model():
    

    from keras.layers import (ELU, Activation, BatchNormalization, Dense, Dropout,
                          InputLayer, Flatten, Reshape, LSTM, TimeDistributed)
    from keras.models import Sequential
    from keras.utils import plot_model

    from PIL import Image

    """ ----------------------------- VANILLA ARTIFICIAL NEURAL NETWORK ----------------------------- """
    # model = Sequential()
    # model.add(InputLayer(input_shape=(1,)))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(60, activation='relu'))
    # model.add(Reshape((10, 6)))
    # model.add(Dense(6, activation='softmax'))

    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    model = Sequential()
    model.add(LSTM(100, input_shape=(10,6), return_sequences=True))
    model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(250, return_sequences=True))
    model.add(LSTM(125, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dropout(rate=0.25))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # plot_model(model, 'model.png', show_layer_names=True, show_shapes=True)
    # Image.open('model.png').show()
    return model


def extract_rules(dictionary):
    for val in dictionary.values():
        if isinstance(val, dict):
            yield from extract_rules(val)
        else:
            yield val


def start_RL(gen_pars, rl_pars):

    model = create_model()

    choices = list(gen_pars.get('variables')) + list(gen_pars.get('constants'))

    onehot = LabelBinarizer(sparse_output=False)
    choices_OH = onehot.fit_transform(choices)

    gen_pars['choices'] = {key: value for (key, value) in enumerate(choices_OH)}

    gen_pars['num_choices'] = len(choices)

    gen_pars['rules'] = {'X': {1: '     ', 2: '     '}}

    rules = list(extract_rules(gen_pars['rules']))

    rule_vec = []

    rows = params.get('num_choices')
    columns = len(''.join(rules))
    reward_table = np.zeros((rows, columns))

    for i in range(rl_pars.get('iterations')):
        rl_pars['eps'] *= rl_pars['decay']

        if i % 100 == 0:
            sys.stdout.write('Iteration {} of {} \n'.format(
                i+1, rl_pars.get('iterations')))

        done = False

        while not done:
            if np.random.random() < rl_params.get('eps'):
                rule_vec_chars = ''.join([np.random.choice(choices) for _ in range(10)])
            else:
                rule_vec_chars = ''.join([np.random.choice(choices) for _ in range(10)])
                # rule_vec += [model.predict()]
                # pass

            rule_vec_OH = onehot.transform(list(rule_vec_chars))[np.newaxis, :, :]

            rule_1 = (rule_vec_chars[:5])
            rule_2 = (rule_vec_chars[5:])
            gen_pars['rules'] = {
                'X': {
                    1: rule_1,
                    2: rule_2,
                }}

            indi = Creature(gen_pars)

            print(indi.l_string)

            model_output = np.asarray(indi.fitness).reshape(-1,1)
            
            model_input = rule_vec_OH
            
            model.fit(model_input, model_output, epochs=1, verbose=2)

            # prediction = model.predict(np.asarray(indi.fitness * 0.9).reshape(-1,1))

            # row_max = prediction.max(axis=1, keepdims=True)

            # prediction[:] = np.where(prediction == row_max, 1, 0)

            # prediction_chars = np.array([onehot.inverse_transform(vec) for vec in prediction])

            # prediction_chars = prediction_chars[0]

            # prediction_chars = ''.join(prediction_chars)

            # rule_1 = (prediction_chars[:5])
            # rule_2 = (prediction_chars[5:])
            # gen_pars['rules'] = {
            #     'X': {
            #         1: rule_1,
            #         2: rule_2,
            #     }}

            # indi_2 = Creature(gen_pars)

            # print('\nPredicted chars: \t' + prediction_chars)
            # print('Predicted fitness: \t' + str(indi_2.fitness))
            # # print('\nInput chars: \t' + rule_vec_chars)
            # print('Input fitness: \t' + str(indi.fitness * 0.9))

            # print()

            predict_vec = onehot.transform(['F','F','F','F','F','F','F','F','F','F',])[np.newaxis, :, :]

            print(model.predict(predict_vec))


def testRNN(gen_pars):
    from Tools.Gen_Tools import open_file

    from keras.callbacks import EarlyStopping

    es = EarlyStopping(monitor='loss', verbose=1, patience=2)

    model = create_model()
    data = open_file()

    data = data[['Rules', 'Fitness']]

    data['Rules'] = data['Rules'].str.join('')
    data = data[data['Rules'].map(len) == 10]

    choices = list(gen_pars.get('variables')) + list(gen_pars.get('constants'))

    binar = LabelBinarizer(sparse_output=False)
    binar.fit_transform(choices)

    # data['Rules'] = data['Rules'].map(
    data['Rules'] = data['Rules'].map(list).map(binar.transform)

    X = np.concatenate(data['Rules'].to_numpy())
    X = X.reshape(-1,10,6)

    scaler = StandardScaler()
    Y = scaler.fit_transform(data[['Fitness']])

    idx = int(len(X) * 0.9)

    X_train = X[:5000]
    Y_train = Y[:5000]

    X_test = X[5001:10000]
    Y_test = Y[5001:10000]

    model.fit(X_train, Y_train, epochs=1, verbose=1, callbacks=[es])

    print(model.evaluate(X_test, Y_test))

    for i in np.random.randint(0, 5000, 5):
        print('L-string: \t {} \t Predicted area: \t {} \t Actual area: \t {}'.format(
            X_test[i],
            scaler.inverse_transform(model.predict(X_test[i])), 
            scaler.inverse_transform(Y_test[i])
            ))


if __name__ == "__main__":
    params = {
        'chars': 500,
        'recurs': 5,
        'variables': 'X',
        'constants': 'F+-[]',
        'axiom': 'FX',
        'length': 1.0,
        'angle': 25,
        'prune': False,
        'pairwise': True,
        'rule_length': 5,
    }

    rl_params = {
        'y': 0.95,
        'eps': 0.5,
        'decay': 0.999,
        'iterations': 1000
    }

    testRNN(params)
    # start_RL(params, rl_params)
