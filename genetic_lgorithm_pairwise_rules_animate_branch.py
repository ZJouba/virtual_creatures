import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from datetime import datetime
from functools import partial
from itertools import chain, repeat
from math import ceil
import curses

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate

from CreatureTools_n import B_Creature


def genPop(params, predef_rules=None):

    np.random.seed()

    if not predef_rules:
        choices = ['F',
                   '+',
                   '-',
                   'X',
                   '[',
                   ']',
                   ]
        rule_A = ''.join([np.random.choice(choices)
                          for _ in range(4)])
        rule_B = ''.join([np.random.choice(choices)
                          for _ in range(5)])
        params['rules'] = {'X': {1: rule_A, 2: rule_B}}
    else:
        if params.get('pairwise'):
            params['rules'] = {'X': {
                1: predef_rules[0],
                2: predef_rules[1]
            }}
        else:
            params['rules'] = {'X': {
                1: predef_rules,
                2: predef_rules
            }}

    c = B_Creature(params)
    a = (
        c.l_string,
        c.coords,
        c.area,
        c.bounds,
        c.F,
        c.perF,
        c.perP,
        c.perM,
        c.perB,
        c.perN,
        c.perX,
        c.maxF,
        c.maxP,
        c.maxM,
        c.avgF,
        c.avgP,
        c.avgM,
        c.angle,
        c.rules,
        c.lines,
        c.ratio,
        c.comp,
        c.fitness,
        # c.l_string,
        # c.area,
        # c.rules,
        # c.ratio,
    )

    return list(a)


def firstRun(iter, params):
    population = []
    population = [[
        'L-string',
        'Coordinates',
        'Area',
        'Bounding Coordinates',
        'No. of F',
        '% of F',
        '% of +',
        '% of -',
        '% of [',
        '% of ]',
        '% of X',
        'Longest F sequence',
        'Longest + sequence',
        'Longest - sequence',
        'Average chars between Fs',
        'Average chars between +s',
        'Average chars between -s',
        'Angle',
        'Rules',
        'Lines',
        'Rule ratio',
        'Compactness',
        'Fitness'
        # 'L-string',
        # 'Area',
        # 'Rules',
        # 'Rule ratio',
    ]]

    with mp.Pool(num_cores) as pool:
        result = list(
            tqdm.tqdm(pool.imap(genPop, repeat(params, 100)), total=100))
        population = population + result

    return population


def selection(population):

    np.random.seed()

    choices = ['F',
               '+',
               '-',
               'X',
               '[',
               ']',
               ]

    elite = 2
    random = 0.05
    mutation = 0.1

    if params.get('pairwise'):
        total = len(population)
    else:
        total = len(population)/2

    next_gen = []

    if params.get('pairwise'):
        total_fitness = sum(
            population['Area A'].values + population['Area B'].values)
        probabilities = pd.Series(
            (population['Area A'].values + population['Area B'].values)/total_fitness)

        """ ELITE """
        for i in range(elite):
            next_gen.append(list(population[['Rule A', 'Rule B']].iloc[i]))

        """ RANDOM """
        random_no = int(total * random)
        for _ in range(random_no):
            next_gen.append([
                (''.join([np.random.choice(choices) for _ in range(5)])),
                (''.join([np.random.choice(choices) for _ in range(5)])),
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
                mutatee_1[i] = np.random.choice(choices)
                mutatee_2[j] = np.random.choice(choices)
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
        total_fitness = sum(population['Area'].values)
        probabilities = pd.Series((population['Area'].values)/total_fitness)

        """ ELITE """
        for i in range(elite):
            next_gen.append(population['Rule'].iloc[i])

        """ RANDOM """
        random_no = int(total * random)
        for _ in range(random_no):
            next_gen.append(
                (''.join([np.random.choice(choices) for _ in range(5)]))
            )

        """ MUTATION """
        mut_no = int(total * mutation)
        for i in range(mut_no):
            mutatee = list(
                population['Rule'].iloc[np.random.randint(0, total)])
            index = np.random.randint(low=0,
                                      high=len(mutatee), size=int(len(mutatee)*0.2))
            for i in index:
                mutatee[i] = np.random.choice(choices)

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

    curr_dir = os.path.dirname(__file__)

    manager = mp.Manager()

    global params, num_cores

    best_area = manager.list([[0, 0]])

    num_cores = mp.cpu_count() - 2

    fig, ax = plt.subplots()
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid()
    line, = plt.plot(best_area[0][0], best_area[0][1], 'r-')

    plot_proc = mp.Process(target=plotting, args=(fig, line, best_area,))

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
    }

    pop_size = [100]

    for pop in pop_size:
        population = firstRun(pop, params)

        sys.stdout.write('\nDone! Creating dataframe \n')

        population = pd.DataFrame(population[1:], columns=population[0])

        population.sort_values(by='Fitness', ascending=False, inplace=True)

        population.reset_index(inplace=True, drop=True)

        if params.get('pairwise'):
            rule_frame = population['Rules'].apply(pd.Series)
            rule_frame.columns = ['Rule A', 'Rule B']
            rule_frame[['Area A', 'Area B']
                       ] = population['Rule ratio'].apply(pd.Series)

        else:
            rule_frame = pd.DataFrame(
                population['Rules'].apply(pd.Series).values.ravel())
            rule_frame.columns = ['Rule']
            rule_frame['Area'] = population['Rule ratio'].apply(
                pd.Series).values.ravel()
            rule_frame.sort_values(by=['Area'], ascending=False, inplace=True)

        sys.stdout.write('Done! \n')

        max_area = params.get('chars')

        i = 0

        top_gens = [list(population.columns)]
        top_gens.append(list(population.iloc[0, :]))

        all_gens = [list(population.columns)]
        all_gens = all_gens + population.values.tolist()

        stdscr = curses.initscr()
        stdscr.refresh()

        plot_proc.start()

        max_area = 1365  # see write-book for formula

        stop_crit = False
        patience = 200
        top_log = population['Area'].iloc[0]
        # tolerance = 1e-4
        teller = 0

        while not stop_crit:  # > tolerance:

            if os.path.exists(os.path.join(curr_dir, 'kill.txt')):
                curses.endwin()
                sys.stdout.write('Manually killed')
                break

            stdscr.addstr(0, 0, 'Iteration: {}\r'.format(i))
            stdscr.refresh()

            result_list = []

            new_gen = selection(rule_frame)

            with mp.Pool(num_cores) as pool:
                func = partial(genPop, params)
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

            population.sort_values(by='Fitness', ascending=False, inplace=True)

            if params.get('pairwise'):
                rule_frame = population['Rules'].apply(pd.Series)
                rule_frame.columns = ['Rule A', 'Rule B']
                rule_frame[['Area A', 'Area B']
                           ] = population['Rule ratio'].apply(pd.Series)

            else:
                rule_frame = pd.DataFrame(
                    population['Rules'].apply(pd.Series).values.ravel())
                rule_frame.columns = ['Rule']
                rule_frame['Area'] = population['Rule ratio'].apply(
                    pd.Series).values.ravel()
                rule_frame.sort_values(
                    by=['Area'], ascending=False, inplace=True)

            best_area.append([i+1, population['Fitness'].iloc[0]])

            top_gens.append(list(population.iloc[0, :]))
            all_gens = all_gens + population.values.tolist()

            if population['Fitness'].iloc[0] > top_log:
                top_log = population['Fitness'].iloc[0]
                teller = 0
            else:
                teller += 1
                # if (teller >= patience):
                #     if (0.9 * max_area) < top_log < (1.1 * max_area):
                #         stop_crit = True
                #         curses.endwin()
                #         sys.stdout.write('Stopping criteria reached!')
                if teller >= patience:
                    stop_crit = True
                    curses.endwin()
                    sys.stdout.write('Stopping criteria reached!')

            i += 1

        plot_proc.join()
        plot_proc.terminate()

        top_gens = pd.DataFrame(top_gens[1:], columns=top_gens[0])
        all_gens = pd.DataFrame(all_gens[1:], columns=all_gens[0])

        now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
        top_frame = os.path.join(
            curr_dir, 'CSVs/GA_top ' + now + '.p')
        final_frame = os.path.join(
            curr_dir, 'CSVs/GA_final_pop ' + now + '.p')
        all_frame = os.path.join(
            curr_dir, 'CSVs/GA_all ' + now + '.p')

        pickle.dump(top_gens, open(top_frame, 'wb'))
        pickle.dump(population, open(final_frame, 'wb'))
        pickle.dump(all_gens, open(all_frame, 'wb'))
        top_gens.to_csv(os.path.join(
            curr_dir, 'CSVs/generations ' + now + '.csv'))
        population.to_csv(os.path.join(
            curr_dir, 'CSVs/final_pop ' + now + '.csv'))
