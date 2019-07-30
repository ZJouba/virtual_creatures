# import plaidml.keras
# plaidml.keras.install_backend()
# import plaidml.keras.backend

import sys
import random
import gym

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt

from Tools.Classes import Creature

def create_model(input_shape, output_shape):
    from keras.layers import (ELU, Activation, BatchNormalization, Dense, Dropout,
                          InputLayer, Flatten, Reshape, LSTM, TimeDistributed, LeakyReLU)
    from keras.models import Sequential
    from keras.optimizers import Adam

    """ ----------------------------- ARTIFICIAL NEURAL NETWORK ----------------------------- """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(200))
    model.add(LeakyReLU())
    model.add(Dense(50))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.2))
    model.add(Dense(output_shape, activation='softmax'))

    adam = Adam(lr=1e-4)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    return model


def start_RL(gen_pars, rl_pars):

    input_shape = rl_pars.get('input_shape')
    gamma = rl_pars.get('gamma')
    epsilon = rl_pars.get('eps')
    decay = rl_pars.get('decay')

    iteration = 0
    iterations = rl_params.get('iterations')

    frequency = iterations // (rl_pars.get('verbosity') * 60)

    action_space = list(gen_pars.get('variables')) + list(gen_pars.get('constants'))


    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = r[t] + running_add * (gamma**t)  # belman equation
            discounted_r[t] = running_add
        return discounted_r


    def discount_n_standardise(r):
        disc_rewards = discount_rewards(r)
        scaler = StandardScaler()
        disc_rewards = scaler.fit_transform(disc_rewards.reshape(-1,1)).reshape(-1,)

        return disc_rewards

    model = create_model(input_shape, len(action_space))

    cum_eps_reward = np.zeros(iterations+1)
    losses = np.zeros(iterations+1)
    reward_sum = 0

    input_size = gen_pars.get('rule_length')
    xs = np.zeros((input_size,input_size))
    ys = np.zeros((input_size, 1))
    rs = np.zeros(input_shape)
    
    k = 0

    rules = []

    while iteration < (iterations+1):
        buffer_x = []
        buffer_y = []
        buffer_r = []

        epsilon *= decay

        while len(buffer_x) < rl_params.get('buffer_size'):

            while k < (input_size):
                # Get current state
                xs[k] = np.identity(input_size)[k]

                # Predict action given current state
                probabilities = model.predict(xs[k][None,:])

                if np.random.random() < epsilon:
                    # action_choice = np.random.choice(len(action_space), p=probabilities[0])
                    action_choice = np.random.randint(0, len(action_space))
                else:
                    action_choice = np.argmax(probabilities)

                action = action_space[action_choice]
                rules.append(action)
                ys[k] = action_choice

                rs[k] = 0

                buffer_x.append(xs[k])
                buffer_y.append(ys[k])

                k += 1  

            """ ---------------------------------------------------------------------------------------------------------------------------------------------
            PAIRWISE IS DIFFICULT TO TRAIN NEURAL NETWORK ON 
            ---------------------------------------------------------------------------------------------------------------------------------------------"""
            rules = ''.join(rules)
            # rule_1 = (rules[:5])
            # rule_2 = (rules[5:])
            gen_pars['rules'] = {
                'X': {
                    1: rules,
                    2: rules,
                }}

            indi = Creature(gen_pars)

            reward = getattr(indi, gen_pars.get('fitness_metric'))
            rs[k-1] = reward
            cum_eps_reward[iteration] = reward

            # Discount rewards - distribute single reward to all actions so each action has an associated reward
            rs = discount_n_standardise(rs)
            buffer_r.extend(rs)

            k = 0
            rules = []
            input_size = gen_pars.get('rule_length')
            xs = np.zeros((input_size,input_size))
            ys = np.zeros((input_size, 1))
            rs = np.zeros(input_shape)

        model.fit(
            np.asarray(buffer_x),
            np.asarray(buffer_y), 
            sample_weight=np.asarray(buffer_r),
            epochs=1, 
            verbose=0,
            shuffle=True
        )

        losses[iteration] = model.evaluate(
            np.asarray(buffer_x),
            np.asarray(buffer_y),
            sample_weight=np.asarray(buffer_r),
            verbose=0,
        )

        if (iteration % frequency == 0) or iteration == iterations:
            ave_reward = np.mean(cum_eps_reward[max(0, iteration-frequency):iteration])
            ave_loss = np.mean(losses[max(0, iteration-frequency):iteration])
            print('Iteration {0:d} of {1:d}\t Average Loss: {2:.2f}\t Average Reward: {3:.2f}'.format(
                iteration,
                iterations,
                ave_loss,
                ave_reward,
            ))
        
        iteration += 1
        
          
    window = 20
    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].plot(losses)
    ax[0].plot(np.convolve(losses, np.ones((window,))/window, mode='same'))
    ax[0].set_title('Losses')

    ax[1].plot(cum_eps_reward)
    ax[1].plot(np.convolve(cum_eps_reward, np.ones((window,))/window, mode='same'))
    ax[1].set_title('Rewards')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    final_rule = []
    for i in range(input_size):
        probabilities = model.predict(
            np.identity(input_size)[i][None,:]
            )
        action_choice = np.argmax(probabilities)
        action = action_space[action_choice]
        final_rule.append(action)

    print('\nNetwork test - Print best rule \t->\t' + ''.join(final_rule))


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
        'fitness_metric': 'area'
    }

    rl_params = {
        'gamma': 0.99,
        'eps': 0.5,
        'decay': 0.999,
        'iterations': 100,
        'input_shape': (params.get('rule_length'),),
        'verbosity': 1,                                   # 1 low -> 10 high
        'buffer_size': 100,
    }

    # testRNN(params)

    start_RL(params, rl_params)
