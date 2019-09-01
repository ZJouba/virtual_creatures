import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import inspect
import gc
from datetime import datetime
from functools import partial
from itertools import chain, repeat
from math import ceil, degrees
import curses

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

        map_dict = {'F': np.random.choice(['Y', 'N'])}

        joints_A = ' '.join([map_dict.get(c, '') for c in rule_A])
        joints_B = ' '.join([map_dict.get(c, '') for c in rule_B])
        GA_params['joints'] = {'X': {1: joints_A, 2: joints_B}}
    else:
        if GA_params.get('pairwise'):
            GA_params['rules'] = {'X': {
                1: predef_rules[0],
                2: predef_rules[1]
            }}
            GA_params['joints'] = {
                'X': {1: predef_rules[4], 2: predef_rules[5]}}
        else:
            GA_params['rules'] = {'X': {
                1: predef_rules,
                2: predef_rules
            }}
            GA_params['joints'] = {
                'X': {1: predef_rules[4], 2: predef_rules[5]}}

        if predef_rules[2] == 'random':
            GA_params['angle'] = 'random'
        else:
            GA_params['angle'] = degrees(predef_rules[3])

    # if GA_params.get('angle') == 'random':
    #     GA_params['angle'] = np.random.randint(0, 90)

    # for _ in range(100):
    c = Creature(GA_params)

    # done = True
    # while done:
    #     try:
    #         c = Creature(GA_params)
    #         done = False
    #     except:
    #         pass

    if listed:
        return list(c.__dict__.keys())
    else:
        return list(c.__dict__.values())


def firstRun(iter, GA_params):

    for _ in range(2):
        init_creature = genPop(GA_params, listed=True)

    population = [init_creature]

    if num_cores > 1:
        try:
            with mp.Pool(num_cores) as pool:
                result = list(
                    tqdm.tqdm(pool.imap(genPop, repeat(GA_params, iter)), total=iter))
                population = population + result
        except:
            traceback.print_exc()
    else:
        for _ in range(iter):
            result = list(genPop(GA_params))
            population.append(result)

    return population


def selection(population, iter):

    global probabilities, random, decay

    np.random.seed()

    if iter == 0:
        decay = 0.95
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
            next_gen.append(
                list(population.iloc[i]))

        """ RANDOM """
        random_no = int(total * random)

        map_dict = {'F': np.random.choice(['Y', 'N'])}

        for _ in range(random_no):

            rule_A = ''.join([np.random.choice(GA_params.get('choices'))
                              for _ in range(GA_params.get('rule_length'))])
            rule_B=''.join([np.random.choice(GA_params.get('choices'))
                                for _ in range(GA_params.get('rule_length'))])

            joints_A = ' '.join([map_dict.get(c, '') for c in rule_A])
            joints_B = ' '.join([map_dict.get(c, '') for c in rule_B])

            next_gen.append([
                rule_A,
                rule_B,
                'random',
                0,
                joints_A,
                joints_B,
            ])

        """ MUTATION """
        mut_no = int(total * mutation)
        for i in range(mut_no):

            ind = np.random.randint(0, total)

            mutatee_1, mutatee_2 = population[[
                'Rule A', 'Rule B']].iloc[ind]
            mutatee_1, mutatee_2 = list(mutatee_1), list(mutatee_2)
            index_1 = np.random.randint(low=0,
                                        high=len(mutatee_1), size=int(len(mutatee_1)*0.2))
            index_2 = np.random.randint(low=0,
                                        high=len(mutatee_2), size=int(len(mutatee_2)*0.2))

            joint_A = list(population['Joint A'].iloc[ind])
            joint_B = list(population['Joint B'].iloc[ind])
            while len(joint_A) < 5:
                joint_A.insert(0, ' ')
            while len(joint_B) < 5:
                joint_B.insert(0, ' ')

            for i, j in zip(index_1, index_2):
                char_1 = np.random.choice(GA_params.get('choices'))
                if char_1 == 'F':
                    joint_A[i] = np.random.choice(['Y', 'N'])
                mutatee_1[i] = char_1

                char_2 = np.random.choice(GA_params.get('choices'))
                if char_2 == 'F':
                    joint_B[j] = np.random.choice(['Y', 'N'])
                mutatee_2[j] = np.random.choice(GA_params.get('choices'))

            next_gen.append([
                ''.join(mutatee_1),
                ''.join(mutatee_2),
                'random',
                0,
                ''.join(joint_A),
                ''.join(joint_B),
            ])

        """ CROSSOVER """
        while len(next_gen) < total:

            ind_1 = np.random.choice(
                population.index, p=probabilities)
            ind_2 = np.random.choice(
                population.index, p=probabilities)

            parent_1_A, parent_1_B = population[[
                'Rule A', 'Rule B']].iloc[ind_1]
            parent_2_A, parent_2_B = population[[
                'Rule A', 'Rule B']].iloc[ind_2]

            joint_1_A, joint_1_B = population[[
                'Joint A', 'Joint B']].iloc[ind_1]
            joint_2_A, joint_2_B = population[[
                'Joint A', 'Joint B']].iloc[ind_2]

            index_1 = np.random.randint(0, len(parent_1_A))
            index_2 = np.random.randint(0, len(parent_2_A))

            child_1 = parent_1_A[:index_1] + parent_2_A[index_1:]
            child_2 = parent_1_B[:index_2] + parent_2_B[index_2:]

            joint_1 = joint_1_A[:index_1] + joint_2_A[index_1:]
            joint_2 = joint_1_B[:index_2] + joint_2_B[index_2:]

            next_gen.append([
                child_1, child_2, 'random', 0, joint_1, joint_2
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
            index_values = np.random.randint(low=0,
                                      high=len(mutatee), size=int(len(mutatee)*0.2))
            for i in index_values:
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


def genRuleFrame(population):
    if GA_params.get('pairwise'):
        rule_frame = population['Rules'].apply(pd.Series)
        rule_frame.columns = ['Rule A', 'Rule B']
        rule_frame[[GA_params.get(
            'fitness_metric')]] = population[GA_params.get(
                'fitness_metric')].apply(pd.Series)
        rule_frame['Angle'] = population['Angle'].apply(pd.Series)
        rule_frame[['Joint A', 'Joint B']] = population['Joints'].apply(
            lambda x: list(x.get('X').values())).apply(pd.Series)
    else:
        rule_frame = pd.DataFrame(
            population['Rules'].apply(pd.Series).values.ravel())
        rule_frame.columns = ['Rule']
        rule_frame['Joint'] = population['Joints'].apply(
            lambda x: list(x.get('X').values())).apply(pd.Series).values.ravel()
        rule_frame['Ratio'] = population['Ratio'].apply(
            pd.Series).values.ravel()
        rule_frame['Angle'] = population['Angle'].apply(pd.Series)
        rule_frame.sort_values(by=['Ratio'], ascending=False, inplace=True)

    return rule_frame


def plotter(frame, line, best_area, value_arr):
    data = np.array(best_area)

    line.set_data(data[:, 0], data[:, 1])
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.relim()
    ax.autoscale()
    return line,


def plotting(fig, line, best_area):
    value_arr = np.empty(2, dtype=object)
    ani = FuncAnimation(fig, plotter, fargs=(
        line, best_area, value_arr, ), interval=1000, save_count=1)
    if len(best_area) > 100:
        ani.event_source.interval = 500

    plt.show()


if __name__ == "__main__":

    test_arr = np.empty(2, dtype=object)
    manager = mp.Manager()

    global GA_params, num_cores

    num_cores = 2  # mp.cpu_count() - 2

    GA_params = {
        'chars': 500,
        'recurs': 3,
        'variables': 'X',
        'constants': 'F+-[]_',
        'axiom': 'FX',
        'length': 1.0,
        'angle': 'random',  # 'random' 'value'
        'learnable': True,
        'prune': False,
        'pairwise': True,
        'achievement': 'Maximum', # 'Best', 'Maximum'
        'rule_length': 4,
        'fitness_metric': 'Area',
        'patience': 5,
        'shape': 'patches',  # 'circle' 'square' 'rainbow' 'triangle' 'patches' 'uniform'
        'richness': 'abundant',  # 'scarce' 'common' 'abundant'
        'scale': 'small',  # 'small' 'medium' 'large'
    }

    use_curses = True
    total_achievable = 0

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

    if not env.patches:
        total_achievable = 0
    else:
        for p in env.patches:
            total_achievable += p.area

    pickle.dump(env, open(env_path, 'wb'))

    GA_params['env'] = env

    # fig, ax = plt.subplots()

    # for p in env.patches:
    #     ax.add_patch(PolygonPatch(p))
    # ax.plot(0, 0, 'xr')
    # plt.autoscale()
    # plt.show()

    GA_params['choices'] = list(GA_params.get(
        'variables') + GA_params.get('constants'))

    best_area = manager.list()
    best_area.append((0,0))

    fig, ax = plt.subplots()
    ax.set_xlabel('Generation')
    ax.set_ylabel(GA_params.get('fitness_metric'))
    ax.grid()
    if total_achievable > 0:
        ax.axhline(total_achievable, linewidth=4, color='grey')
        plt.text(0, total_achievable+0.1, 'Total available area')
    line, = plt.plot(0, 0, 'r-', alpha=0.8)

    plot_proc = mp.Process(target=plotting, args=(fig, line, best_area,))

    pop_size = [10]

    for pop in pop_size:
        population = firstRun(pop, GA_params)

        sys.stdout.write('\nDone! Creating dataframe \n')

        population = pd.DataFrame(population[1:], columns=population[0])

        population.sort_values(by=GA_params.get(
            'fitness_metric'), ascending=False, inplace=True)

        population.reset_index(inplace=True, drop=True)

        rule_frame = genRuleFrame(population)

        sys.stdout.write('Done! \n')

        i = 0

        top_gens = [list(population.columns)]
        top_gens.append(list(population.iloc[0, :]))

        all_gens = [list(population.columns)]
        all_gens = all_gens + population.values.tolist()

        if use_curses:
            stdscr = curses.initscr()
            stdscr.refresh()

        plot_proc.start()

        stop_crit = False
        top_log = population[GA_params.get('fitness_metric')].iloc[0]
        teller = 0

        dur = 0
        while not stop_crit:

            if os.path.exists(os.path.join(curr_dir, 'kill.txt')):
                if use_curses:
                    curses.endwin()
                sys.stdout.write('Manually killed')
                break

            if use_curses:
                stdscr.addstr(0, 0, 'Iteration: {}\r'.format(i))
                stdscr.refresh()

            new_gen = selection(rule_frame, i)
            start = time.time()

            if num_cores > 1:
                try:
                    """ -------------- MULTIPROCESSING ---------------------- """
                    with mp.Pool(num_cores) as pool:
                        func = partial(genPop, GA_params)
                        """ WITH PROGRESSBARS """
                        # result = list(tqdm.tqdm(pool.imap_unordered(func, new_gen), total=len(new_gen), file=sys.stdout))
                        """ WITHOUT PROGRESSBARS """
                        if use_curses:
                            stdscr.addstr(1, 0, 'Busy...')
                            stdscr.clrtoeol()
                            stdscr.refresh()
                        result = list(pool.imap_unordered(func, new_gen))
                        if use_curses:
                            stdscr.addstr(1, 0, 'Done')
                            stdscr.clrtoeol()
                            stdscr.refresh()
                except:
                    traceback.print_exc()

            else:
                """ -------------- NORMAL ---------------------- """
                result = []
                for dna in new_gen:
                    if use_curses:
                        stdscr.addstr(1, 0, 'Busy...')
                        stdscr.clrtoeol()
                        stdscr.refresh()
                    result.append(list(genPop(GA_params, dna)))
                    if use_curses:
                        stdscr.addstr(1, 0, 'Done')
                        stdscr.clrtoeol()
                        stdscr.refresh()

            end = time.time()

            dur += (end - start)

            avg = dur/(i+1)

            if use_curses:
                stdscr.addstr(
                    2, 0, 'Running average class time: \t {:.5f} second'.format(avg))

            if num_cores > 1:
                pool.join()

            population.iloc[:, :] = result

            population.sort_values(by=GA_params.get(
                'fitness_metric'), ascending=False, inplace=True)

            rule_frame = genRuleFrame(population)

            best_area.append(
                (i+1, population[GA_params.get('fitness_metric')].iloc[0]))

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
                if GA_params['achievement'] == 'Maximum':
                    if (total_achievable*0.8) >= np.amax(top_log) >= total_achievable:
                        teller += 1
                        if teller >= GA_params.get('patience'):
                            stop_crit = True
                            if use_curses:
                                curses.endwin()
                            sys.stdout.write('Stopping criteria reached!')
                else:
                    teller += 1
                    if teller >= GA_params.get('patience'):
                        stop_crit = True
                        if use_curses:
                            curses.endwin()
                        sys.stdout.write('Stopping criteria reached!')

            i += 1
            gc.collect()

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
