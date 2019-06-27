from CreatureTools_n import Creature
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from datetime import datetime
import sys
import time
import pickle
from itertools import chain
import random
from math import ceil
import matplotlib.pyplot as plt


def initial_Pop():
    choices = [
        'F',
        '+',
        '-',
    ]

    proba1 = np.random.uniform(0, 1)
    proba2 = 1 - proba1

    rule1 = ''.join([np.random.choice(choices)
                     for _ in range(5)]) + 'X'
    rule2 = ''.join([np.random.choice(choices)
                     for _ in range(5)]) + 'X'

    params = {
        'num_char': 100,
        'variables': 'X',
        'constants': 'F+-',
        'axiom': 'FX',
        'rules': {
            'X': {
                'options': [
                    rule1,
                    rule2,
                ],
                'probabilities': [proba1, proba2]
            }
        },
        'point': np.array([0, 0]),
        'vector': np.array([0, 1]),
        'length': 1.0,
        'angle': np.random.randint(0, 90)  # random
    }

    c = Creature(params)
    a = (
        c.l_string,
        c.coords,
        c.area,
        c.bounds,
        c.perF,
        c.perP,
        c.perM,
        c.maxF,
        c.maxP,
        c.maxM,
        c.avgF,
        c.avgP,
        c.avgM,
        c.angle,
        c.rules,
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
        'Coordinates',
        'Area',
        'Bounding Coordinates',
        '% of F',
        '% of +',
        '% of -',
        'Longest F sequence',
        'Longest + sequence',
        'Longest - sequence',
        'Average chars between Fs',
        'Average chars between +s',
        'Average chars between -s',
        'Angle',
        'Rules'
    ]]

    with mp.Pool() as pool:

        for i in range(iter):
            progress(i, iter, status='Create initial population')

            np.random.seed()

            results = pool.apply_async(initial_Pop)
            population.append(results.get())

    pool.join()

    return population


def selection(population):

    np.random.seed()

    elite = 0.5
    random = 0

    total = population.shape[0]

    next_gen = []

    """ RANDOM """
    random_no = int(total * random)
    for _ in range(random_no):
        next_gen.append((''.join(np.random.choice(['F', '+', '-'], 501))))

    """ ELITE """
    elite_no = int(total * elite)
    for i in range(elite_no):
        next_gen.append(population.iloc[i, 0])

    """ CROSSOVER """
    while len(next_gen) < total:
        parent1 = np.random.choice(population['L-string'])
        parent2 = np.random.choice(population['L-string'])
        index = np.random.randint(0, len(parent1))
        child = parent1[:index] + parent2[index:]
        next_gen.append(child)

    tmp = []
    for crt in next_gen:
        param = {
            'L-string': crt,
            'length': 1.0,
            'angle': np.random.randint(0, 90)
        }
        c = Creature(param)
        a = (
            c.l_string,
            c.coords,
            c.area,
            c.bounds,
            c.perF,
            c.perP,
            c.perM,
            c.maxF,
            c.maxP,
            c.maxM,
            c.avgF,
            c.avgP,
            c.avgM,
            c.angle,
        )
        tmp.append(a)

    return tmp


if __name__ == "__main__":

    pop_size = [100,200,300,400,500,600,700,800,900,1000]

    population = firstRun(pop_size)

    sys.stdout.write('\nDone! Creating dataframe \n')

    population_frame = pd.DataFrame(population[1:], columns=population[0])
    population_frame.sort_values(by=['Area'], inplace=True)

    sys.stdout.write('Done! \n')

    max_area = 500

    population_frame.drop(['Rules'], axis=1, inplace=True)

    i = 0

    best_area = []
    plt.ion()

    gen_frame = [list(population_frame.columns)]

    for pop in pop_size:
        while population_frame.iloc[0, 2] < max_area:

            gen_frame.append(list(population_frame.iloc[0, :]))

            result_list = []

            num_cores = 6
            num_partitions = 10
            split_frames = np.split(population_frame, num_partitions)
            with mp.Pool(num_cores) as pool:
                result = pool.map(selection, split_frames)
                result_list.append(result)

            pool.join()

            unpacked_results = list(chain(*result_list[0]))

            population_frame.iloc[:, :] = unpacked_results

            sys.stdout.write('Iteration: {}\r'.format(i))

            population_frame.sort_values(
                by=['Area'], ascending=False, inplace=True)

            best_area.append(population_frame.iloc[0, 2])
            plt.clf()
            plt.plot(best_area)
            plt.xlabel('Generation')
            plt.ylabel('Best area')
            plt.pause(0.0001)

            i += 1

        gen_frame = pd.DataFrame(gen_frame[1:], columns=gen_frame[0])

        sys.stdout.write('\nDone! \n')

        curr_dir = os.path.dirname(__file__)

        now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
        file_name = os.path.join(
            curr_dir, 'CSVs/genetic_lgorithm ' + now + '.p')
        file1_name = os.path.join(
            curr_dir, 'CSVs/genetic_lgorithm_genframe ' + now + '.p')
        pic_name = os.path.join(
            curr_dir, 'CSVs/plot_iterations=' + str(i))

        plt.savefig(pic_name)
        pickle.dump(gen_frame, open(file_name, 'wb'))
        pickle.dump(population_frame, open(file1_name, 'wb'))
