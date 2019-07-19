from types import SimpleNamespace

import sys
import random

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer

from Tools.Classes import Creature

def create_model():
    import plaidml.keras
    plaidml.keras.install_backend()
    import plaidml.keras.backend

    from keras.layers import (ELU, Activation, BatchNormalization, Dense, Dropout,
                          InputLayer, Flatten, Reshape)
    from keras.models import Sequential
    from keras.utils import plot_model

    from PIL import Image


    model = Sequential()
    model.add(InputLayer(input_shape=(1,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(600, activation='relu'))
    model.add(Reshape((10, 6,), input_shape=(600,1)))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    plot_model(model, 'model.png', show_layer_names=True, show_shapes=True)
    Image.open('model.png').show()
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

            model_input = np.asarray(indi.fitness).reshape(-1,1)
            
            model_output = rule_vec_OH
            
            model.fit(model_input, model_output, epochs=1, verbose=0)

            predict_vec = onehot.transform(['F','F','F','F','F','F','F','F','F','F',])[np.newaxis, :, :]

            print(model.predict(predict_vec))

            print()


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

    start_RL(params, rl_params)
