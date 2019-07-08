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

from CreatureTools_3 import Creature

def genPop(params, predef_rules=None):
    
    np.random.seed()

    if not predef_rules:
        choices = [ 'F',
                    '+',
                    '-',
                    'X',
                    ]
        rule_A = ''.join([np.random.choice(choices)
                        for _ in range(5)])
        rule_B = ''.join([np.random.choice(choices)
                        for _ in range(5)])
        params['rules'] = {'X': { 1: rule_A, 2:rule_B}}
    else:
        params['rules'] = {'X': { 
            1: predef_rules[0], 
            2: predef_rules[1]
            }}

    c = Creature(params)
    a = (
        c.l_string,
        c.area,
        c.rules,
        c.ratio,
    )

    return list(a)

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))

def firstRun(iter):
    population = []
    population = [[
        'L-string',
        'Area',
        'Rules',
        'Rule ratio',
    ]]

    with mp.Pool(num_cores) as pool:
        result = list(tqdm.tqdm(pool.imap(genPop, repeat(params, iter)), total=iter))
        population = population + result  

    return population

def selection(population):

    np.random.seed()

    choices = [ 'F',
                '+',
                '-',
                'X',
                ]

    random = 0.05
    mutation = 0.1

    total = len(population)

    next_gen = []
    roulette = np.linspace(0, 1, total/2)

    total_fitness = sum(population['Area A'].values + population['Area B'].values)
    probabilities = pd.Series((population['Area A'].values + population['Area B'].values)/total_fitness)

    """ ELITE """
    next_gen.append(list(population.iloc[0,0:2]))
  
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
        mutatee_1, mutatee_2 = population[['Rule A', 'Rule B']].iloc[np.random.randint(0, total), 0:2]
        mutatee_1, mutatee_2 = list(mutatee_1), list(mutatee_2)
        index_1 = np.random.randint(low=0,
                                  high=len(mutatee_1), size=int(len(mutatee_1)*0.2))
        index_2 = np.random.randint(low=0,
                                  high=len(mutatee_2), size=int(len(mutatee_2)*0.2))
        for i,j in zip(index_1, index_2):
            mutatee_1[i] = np.random.choice(['F', '+', '-'])
            mutatee_2[i] = np.random.choice(['F', '+', '-'])
        next_gen.append([
            ''.join(mutatee_1),
            ''.join(mutatee_2),
        ])

    """ CROSSOVER """
    while len(next_gen) < total:
        parent_1_A, parent_1_B  = population.iloc[np.random.choice(population.index, p=probabilities), 0:2]
        parent_2_A, parent_2_B  = population.iloc[np.random.choice(population.index, p=probabilities), 0:2]
        index_1 = np.random.randint(0, len(parent_1_A))
        index_2 = np.random.randint(0, len(parent_2_A))
        child_1 = parent_1_A[:index_1] + parent_2_A[index_1:]
        child_2 = parent_1_B[:index_2] + parent_2_B[index_2:]
        next_gen.append([
            child_1, child_2
        ])

    return next_gen

def plotter(frame, line):
    line.set_data(best_area)
    return line,

def plotting(fig, line):
    ani = FuncAnimation(fig, plotter, fargs=(line,), interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    
    manager = mp.Manager()

    global params, num_cores
    
    best_area = manager.Array('f', [2,1])
    
    num_cores = mp.cpu_count() - 2

    # best_area = np.zeros((2,1))

    fig, ax = plt.subplots()
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best area')
    ax.grid()
    temp = best_area.acquire()
    line, = plt.plot(best_area[0], best_area[1], 'r-')

    plot_proc = mp.Process(target=plotting, args=(fig,line,))
    plot_proc.start()
    

    params = {
        'chars': 500,
        'recurs': 100,
        'variables': 'X',
        'constants': 'F+-',
        'axiom': 'FX',
        'length': 1.0,
        'angle': 25,
        'prune': False,
    }   

    pop_size = [100]

    # for i in range(5):
    #     genPop(params)

    for pop in pop_size:
        population = firstRun(pop)

        sys.stdout.write('\nDone! Creating dataframe \n')

        population = pd.DataFrame(population[1:], columns=population[0])
        population.sort_values(by=['Area'], ascending=False, inplace=True)
        population.reset_index(inplace=True, drop=True)

        rule_frame = population['Rules'].apply(pd.Series)
        rule_frame.columns = ['Rule A', 'Rule B']
        rule_frame[['Area A', 'Area B']] = pd.DataFrame(
            population['Area'].values.reshape(-1,1) * (population['Rule ratio'].apply(pd.Series).values/100)
            )

        sys.stdout.write('Done! \n')
        
        max_area = len(population.iloc[0, 0]) + 0.785

        i = 0
        
        gens = [list(population.columns)]

        stdscr = curses.initscr()
        stdscr.refresh()

        while population.iloc[0, 1] < 5000: #max_area:
            
            stdscr.addstr(0, 0, 'Iteration: {}\r'.format(i))
            stdscr.refresh()

            result_list = []

            gens.append(list(population.iloc[0, :]))

            new_gen = selection(rule_frame)

            with mp.Pool(num_cores) as pool:
                func = partial(genPop, params)
                """ WITH PROGRESSBARS """
                # result = list(tqdm.tqdm(pool.imap_unordered(func, new_gen), total=len(new_gen), file=sys.stdout))
                """ WITHOUT PROGRESSBARS """
                # sys.stdout.write('\nBusy...\r')
                stdscr.addstr(1, 0, 'Busy...')
                stdscr.refresh()
                result = list(pool.imap_unordered(func, new_gen))
                stdscr.addstr(1, 0, 'Done')
                stdscr.refresh()
                
            pool.join()

            population.iloc[:, :] = result

            # sys.stdout.write('Iteration: {}\r'.format(i))
            # sys.stdout.write(tabulate(population.head(1), headers='keys'))

            population.sort_values(
                by=['Area'], ascending=False, inplace=True)

            best_area = np.hstack((best_area, np.array([[i], [population.iloc[0, 1]]])))
            
            # line.set_data(best_area)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.clf()
            # plt.plot(best_area)
            # plt.xlabel('Generation')
            # plt.ylabel('Best area')
            # plt.grid()
            # plt.pause(0.0001)

            i += 1

        curses.endwin()
        sys.stdout.write('Maximum area achieved!')

        gens = pd.DataFrame(gens[1:], columns=gens[0])

        curr_dir = os.path.dirname(__file__)

        now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
        file_name = os.path.join(
            curr_dir, 'CSVs/genetic_lgorithm ' + now + '.p')
        file1_name = os.path.join(
            curr_dir, 'CSVs/genetic_lgorithm_genframe ' + now + '.p')
        pic_name = os.path.join(
            curr_dir, 'CSVs/plot_iterations=' + str(i) + 'pop=' + str(pop))

        plt.savefig(pic_name)
        pickle.dump(gens, open(file_name, 'wb'))
        pickle.dump(population, open(file1_name, 'wb'))
