from CreatureTools_3 import Creature
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
from tabulate import tabulate
import bisect

def genPop():
    
    np.random.seed()

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

    # params = {
    #     'chars': 100,
    #     'variables': 'X',
    #     'constants': 'F+-',
    #     'axiom': 'FX',
    #     'length': 1.0,
    #     'angle': 25
    # }
    
    for i in range(iter):
        progress(i, iter, status='Create initial population')

        np.random.seed()

        population.append(genPop())

    # with mp.Pool() as pool:

    #     for i in range(iter):
    #         progress(i, iter, status='Create initial population')

    #         np.random.seed()

    #         results = pool.apply_async(genPop, args=(params,))
    #         population.append(results.get())

    # pool.join()

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
        parent_1_A, parent_1_B  = np.random.choice(population, p=probabilities)
        parent_2_A, parent_2_B  = np.random.choice(population, p=probabilities)
        index = np.random.randint(0, len(parent1))
        child = parent1[:index] + parent2[index:]
        next_gen.append(child)

    return next_gen

    # tmp = []
    # for crt in next_gen:
    #     param = {
    #         'L-string': crt,
    #         'length': 1.0,
    #         'angle': 45
    #     }
    #     c = Creature(param)
    #     a = (
    #         c.l_string,
    #         c.coords,
    #         c.area,
    #     )
    #     tmp.append(a)

    # return tmp


if __name__ == "__main__":

    global params
    params = {
        'chars': 100,
        'variables': 'X',
        'constants': 'F+-',
        'axiom': 'FX',
        'length': 1.0,
        'angle': 25
    }

    pop_size = [100]

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
        
        max_area = params.get('chars')

        i = 0

        best_area = []
        plt.ion()

        gens = [list(population.columns)]

        while population.iloc[0, 1] < max_area:

            def newGen(crt):
                param = {
                    'L-string': crt,
                    'length': 1.0,
                    'angle': 45
                }
                c = Creature(param)
                a = (
                    c.l_string,
                    c.coords,
                    c.area,
                )
                tmp.append(a)

                return tmp

            result_list = []

            gens.append(list(population.iloc[0, :]))

            # rule_list = list(rule_frame)

            # num_cores = 6
            # num_partitions = 10
            # splits = np.split(populations, num_partitions)
            # with mp.Pool(num_cores) as pool:
            #     result = pool.map(selection, splits)
            #     result_list.append(result)

            # pool.join()

            # result_list = selection(rule_list)
            new_gen = selection(rule_frame)

            with mp.Pool(num_cores) as pool:
                result = pool.map(newGen, new_gen)
                result_list.append(result)

            pool.join()

            # unpacked_results = list(chain(*result_list[0]))

            # populations.iloc[:, :] = unpacked_results

            population.iloc[:, :] = result_list

            sys.stdout.write('Iteration: {}\r'.format(i))

            population.sort_values(
                by=['Area'], ascending=False, inplace=True)

            best_area.append(population.iloc[0, 2])

            plt.clf()
            plt.plot(best_area)
            plt.xlabel('Generation')
            plt.ylabel('Best area')
            plt.grid()

            plt.pause(0.01)

            i += 1

        gens = pd.DataFrame(gens[1:], columns=gens[0])

        # if i == 1000:
        #     sys.stdout.write('\nMaximum iterations reached \n')
        # else:
        #     sys.stdout.write('\nMaximum area achieved \n')

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
