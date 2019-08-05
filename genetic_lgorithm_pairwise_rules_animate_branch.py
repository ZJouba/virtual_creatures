import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import inspect
from datetime import datetime
from functools import partial
from itertools import chain, repeat
from math import ceil
import curses

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate

from Tools.Classes import *


def genPop(GA_params, predef_rules=None, listed=False):

    np.random.seed()

    if not predef_rules:
        rule_A = ''.join([np.random.choice(GA_params.get('choices'))
                          for _ in range(GA_params.get('rule_length'))])
        rule_B = ''.join([np.random.choice(GA_params.get('choices'))
                          for _ in range(GA_params.get('rule_length'))])
        GA_params['rules'] = {'X': {1: rule_A, 2: rule_B}}
    else:
        if GA_params.get('pairwise'):
            GA_params['rules'] = {'X': {
                1: predef_rules[0],
                2: predef_rules[1]
            }}
        else:
            GA_params['rules'] = {'X': {
                1: predef_rules,
                2: predef_rules
            }}

    if GA_params.get('angle') == 'random':
        GA_params['angle'] = np.random.randint(0, 90)

    # for _ in range(100):
    #     c = Creature(GA_params)

    done = True
    while done:
        try:
            c = Creature(GA_params)
            done = False
        except:
            pass

    if listed:
        return list(c.__dict__.keys())
    else:
        return list(c.__dict__.values())


def firstRun(iter, GA_params):

    if GA_params.get('angle') == 'random':
        GA_params['angle'] = np.random.randint(0, 90)
        init_creature = genPop(GA_params, listed=True)
        GA_params['angle'] = 'random'
    else:
        init_creature = genPop(GA_params, listed=True)

    population = [init_creature]

    with mp.Pool(num_cores) as pool:
        result = list(
            tqdm.tqdm(pool.imap(genPop, repeat(GA_params, 100)), total=100))
        population = population + result

    return population


def selection(population, iter):

    global probabilities, random

    np.random.seed()

    if iter == 0:
        decay = 0.85
        random = 0.5
    elif random > 0.05:
        random *= decay

    elite = 2
    mutation = 0.1

    if GA_params.get('pairwise'):
        total = len(population)
    else:
        total = len(population)/2

    next_gen = []

    if GA_params.get('pairwise'):
        # total_fitness = sum(
        #     population['Ratio A'].values + population['Ratio B'].values)
        total_fitness = sum(population[GA_params.get('fitness_metric')].values)
        # probabilities = pd.Series(
        #     (population['Ratio A'].values + population['Ratio B'].values)/total_fitness)
        if total_fitness == 0 and iter == 0:
            probabilities = np.random.dirichlet(
                np.ones(len(population[GA_params.get('fitness_metric')])), )
        elif not total_fitness == 0:
            probabilities = pd.Series(np.divide(
                population[GA_params.get('fitness_metric')].values, total_fitness))

        """ ELITE """
        for i in range(elite):
            next_gen.append(list(population[['Rule A', 'Rule B']].iloc[i]))

        """ RANDOM """
        random_no = int(total * random)
        for _ in range(random_no):
            next_gen.append([
                (''.join([np.random.choice(GA_params.get('choices'))
                          for _ in range(GA_params.get('rule_length'))])),
                (''.join([np.random.choice(GA_params.get('choices'))
                          for _ in range(GA_params.get('rule_length'))])),
            ])

        """ MUTATION """
        mut_no = int(total * mutation)
        for i in range(mut_no):
            mutatee_1, mutatee_2 = population[[
                'Rule A', 'Rule B']].iloc[np.random.randint(0, total)]
            mutatee_1, mutatee_2 = list(mutatee_1), list(mutatee_2)
            index_1 = np.random.randint(low=0,
                                        high=len(mutatee_1), size=int(len(mutatee_1)*0.2))
            index_2 = np.random.randint(low=0,
                                        high=len(mutatee_2), size=int(len(mutatee_2)*0.2))
            for i, j in zip(index_1, index_2):
                mutatee_1[i] = np.random.choice(GA_params.get('choices'))
                mutatee_2[j] = np.random.choice(GA_params.get('choices'))
            next_gen.append([
                ''.join(mutatee_1),
                ''.join(mutatee_2),
            ])

        """ CROSSOVER """
        while len(next_gen) < total:
            parent_1_A, parent_1_B = population[[
                'Rule A', 'Rule B']].iloc[np.random.choice(
                    population.index, p=probabilities)]
            parent_2_A, parent_2_B = population[[
                'Rule A', 'Rule B']].iloc[np.random.choice(
                    population.index, p=probabilities)]
            index_1 = np.random.randint(0, len(parent_1_A))
            index_2 = np.random.randint(0, len(parent_2_A))
            child_1 = parent_1_A[:index_1] + parent_2_A[index_1:]
            child_2 = parent_1_B[:index_2] + parent_2_B[index_2:]
            next_gen.append([
                child_1, child_2
            ])
    else:
        total_fitness = sum(population[GA_params.get('fitness_metric')].values)
        if not total_fitness == 0:
            probabilities = pd.Series(np.divide(
                population[GA_params.get('fitness_metric')].values, total_fitness))
        else:
            probabilities = np.random.dirichlet(
                np.ones(len(population[GA_params.get('fitness_metric')])), )

        """ ELITE """
        for i in range(elite):
            next_gen.append(population['Rule'].iloc[i])

        """ RANDOM """
        random_no = int(total * random)
        for _ in range(random_no):
            next_gen.append(
                (''.join([np.random.choice(GA_params.get('choices'))
                          for _ in range(GA_params.get('rule_length'))]))
            )

        """ MUTATION """
        mut_no = int(total * mutation)
        for i in range(mut_no):
            mutatee = list(
                population['Rule'].iloc[np.random.randint(0, total)])
            index = np.random.randint(low=0,
                                      high=len(mutatee), size=int(len(mutatee)*0.2))
            for i in index:
                mutatee[i] = np.random.choice(GA_params.get('choices'))

            next_gen.append(
                ''.join(mutatee)
            )

        """ CROSSOVER """
        while len(next_gen) < total:
            parent_1 = population['Rule'].iloc[np.random.choice(
                population.index, p=probabilities)]
            parent_2 = population['Rule'].iloc[np.random.choice(
                population.index, p=probabilities)]
            index = np.random.randint(0, len(parent_1))
            child = parent_1[:index] + parent_2[index:]
            next_gen.append(
                child
            )

    return next_gen


def plotter(frame, line, best_area):
    value_arr = np.asarray(best_area)
    try:
        value_arr = value_arr.T
    except:
        pass
    line.set_data(value_arr)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.relim()
    ax.autoscale()
    return line,


def plotting(fig, line, best_area):
    ani = FuncAnimation(fig, plotter, fargs=(
        line, best_area, ), interval=200, save_count=1)
    if len(best_area) > 100:
        ani.event_source.interval = 500
    plt.show()


if __name__ == "__main__":

    manager = mp.Manager()

    global GA_params, num_cores

    GA_params = {
        'chars': 500,
        'recurs': 5,
        'variables': 'X',
        'constants': 'F+-[]_',
        'axiom': 'FX',
        'length': 1.0,
        'angle': 'random',
        'prune': False,
        'pairwise': True,
        'rule_length': 5,
        'fitness_metric': 'Area',
        'patience': 500,
        'shape': 'square',  # 'circle' 'square' 'rainbow' 'triangle' 'patches'
        'richness': 'common',  # 'scarce' 'common' 'abundant'
        'scale': 'small',  # 'small' 'medium' 'large'
    }

    """ -------- PATHS AND FILE NAMES ---------- """
    curr_dir = os.path.dirname(__file__)
    now = datetime.now().strftime('%b %d, %Y @ %H.%M')
    top_frame = os.path.join(
        curr_dir, 'CSVs\\GA_top ' + now + '.p')
    final_frame = os.path.join(
        curr_dir, 'CSVs\\GA_final_pop ' + now + '.p')
    all_frame = os.path.join(
        curr_dir, 'CSVs\\GA_all ' + now + '.p')
    env_path = os.path.join(
        curr_dir, 'CSVs\\Environment ' + now + '.p')

    """ ---------------------------------------- """

    env = Environment(GA_params)

    pickle.dump(env, open(env_path, 'wb'))

    GA_params['env'] = env

    # fig, ax = plt.subplots()

    # for p in env.patches:
    #     ax.add_patch(PolygonPatch(p))

    # plt.autoscale()
    # plt.show()

    GA_params['choices'] = list(GA_params.get(
        'variables') + GA_params.get('constants'))

    best_area = manager.list([[0, 0]])

    num_cores = mp.cpu_count() - 2

    fig, ax = plt.subplots()
    ax.set_xlabel('Generation')
    ax.set_ylabel(GA_params.get('fitness_metric'))
    ax.grid()
    line, = plt.plot(best_area[0][0], best_area[0][1], 'r-')

    plot_proc = mp.Process(target=plotting, args=(fig, line, best_area,))

    pop_size = [200]

    for pop in pop_size:
        population = firstRun(pop, GA_params)

        sys.stdout.write('\nDone! Creating dataframe \n')

        population = pd.DataFrame(population[1:], columns=population[0])

        population.sort_values(by=GA_params.get(
            'fitness_metric'), ascending=False, inplace=True)

        population.reset_index(inplace=True, drop=True)

        if GA_params.get('pairwise'):
            # rule_frame = population['Rules'].apply(pd.Series)
            # rule_frame.columns = ['Rule A', 'Rule B']
            # rule_frame[['Ratio A', 'Ratio B']
            #            ] = population['Ratio'].apply(pd.Series)
            rule_frame = population['Rules'].apply(pd.Series)
            rule_frame.columns = ['Rule A', 'Rule B']
            rule_frame[[GA_params.get(
                'fitness_metric')]] = population[GA_params.get(
                    'fitness_metric')].apply(pd.Series)

        else:
            rule_frame = pd.DataFrame(
                population['Rules'].apply(pd.Series).values.ravel())
            rule_frame.columns = ['Rule']
            rule_frame['Ratio'] = population['Ratio'].apply(
                pd.Series).values.ravel()
            rule_frame.sort_values(by=['Ratio'], ascending=False, inplace=True)

        sys.stdout.write('Done! \n')

        i = 0

        top_gens = [list(population.columns)]
        top_gens.append(list(population.iloc[0, :]))

        all_gens = [list(population.columns)]
        all_gens = all_gens + population.values.tolist()

        stdscr = curses.initscr()
        stdscr.refresh()

        plot_proc.start()

        stop_crit = False
        top_log = population[GA_params.get('fitness_metric')].iloc[0]
        teller = 0

        while not stop_crit:

            if os.path.exists(os.path.join(curr_dir, 'kill.txt')):
                curses.endwin()
                sys.stdout.write('Manually killed')
                break

            stdscr.addstr(0, 0, 'Iteration: {}\r'.format(i))
            stdscr.refresh()

            result_list = []

            new_gen = selection(rule_frame, i)

            with mp.Pool(num_cores) as pool:
                func = partial(genPop, GA_params)
                """ WITH PROGRESSBARS """
                # result = list(tqdm.tqdm(pool.imap_unordered(func, new_gen), total=len(new_gen), file=sys.stdout))
                """ WITHOUT PROGRESSBARS """
                stdscr.addstr(1, 0, 'Busy...')
                stdscr.clrtoeol()
                stdscr.refresh()
                result = list(pool.imap_unordered(func, new_gen))
                stdscr.addstr(1, 0, 'Done')
                stdscr.clrtoeol()
                stdscr.refresh()

            pool.join()

            population.iloc[:, :] = result

            population.sort_values(by=GA_params.get(
                'fitness_metric'), ascending=False, inplace=True)

            if GA_params.get('pairwise'):
                # rule_frame = population['Rules'].apply(pd.Series)
                # rule_frame.columns = ['Rule A', 'Rule B']
                # rule_frame[['Ratio A', 'Ratio B']
                #            ] = population['Ratio'].apply(pd.Series)
                rule_frame = population['Rules'].apply(pd.Series)
                rule_frame.columns = ['Rule A', 'Rule B']
                rule_frame[[GA_params.get(
                    'fitness_metric')]] = population[GA_params.get(
                        'fitness_metric')].apply(pd.Series)

            else:
                rule_frame = pd.DataFrame(
                    population['Rules'].apply(pd.Series).values.ravel())
                rule_frame.columns = ['Rule']
                rule_frame['Ratio'] = population['Ratio'].apply(
                    pd.Series).values.ravel()
                rule_frame.sort_values(
                    by=['Ratio'], ascending=False, inplace=True)

            best_area.append(
                [i+1, population[GA_params.get('fitness_metric')].iloc[0]])

            top_gens.append(list(population.iloc[0, :]))
            all_gens = all_gens + population.values.tolist()

            if i % 10 == 0:
                top_gens_frame = pd.DataFrame(
                    top_gens[1:], columns=top_gens[0])
                all_gens_frame = pd.DataFrame(
                    all_gens[1:], columns=all_gens[0])

                pickle.dump(top_gens_frame, open(top_frame, 'wb'))
                pickle.dump(population, open(final_frame, 'wb'))
                pickle.dump(all_gens_frame, open(all_frame, 'wb'))

                top_gens_frame.to_csv(os.path.join(
                    curr_dir, 'CSVs/generations ' + now + '.csv'))
                population.to_csv(os.path.join(
                    curr_dir, 'CSVs/final_pop ' + now + '.csv'))

            if population[GA_params.get('fitness_metric')].iloc[0] > top_log:
                top_log = population[GA_params.get('fitness_metric')].iloc[0]
                teller = 0
            else:
                teller += 1
                if teller >= GA_params.get('patience'):
                    stop_crit = True
                    curses.endwin()
                    sys.stdout.write('Stopping criteria reached!')

            i += 1

        top_gens_frame = pd.DataFrame(top_gens[1:], columns=top_gens[0])
        all_gens_frame = pd.DataFrame(all_gens[1:], columns=all_gens[0])

        pickle.dump(top_gens_frame, open(top_frame, 'wb'))
        pickle.dump(population, open(final_frame, 'wb'))
        pickle.dump(all_gens_frame, open(all_frame, 'wb'))
        top_gens_frame.to_csv(os.path.join(
            curr_dir, 'CSVs/generations ' + now + '.csv'))
        population.to_csv(os.path.join(
            curr_dir, 'CSVs/final_pop ' + now + '.csv'))

        plot_proc.join()
        plot_proc.terminate()
