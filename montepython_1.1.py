from CreatureTools import Creature
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from tabulate import tabulate
import os
import csv
import pandas as pd


def genGen():
    params = {
        'num_char': 100,
        'variables': 'X',
        'constants': 'F+-',
        'axiom': 'FX',
        'rules': {
            'X': {
                'options': ['+FX', '-FX'],
                'probabilities': [0.5, 0.5]
            }
        },
        'point': np.array([0, 0]),
        'vector': np.array([0, 1]),
        'length': 1.0,
        'angle': 25  # random
    }

    c = Creature(params)
    a = (
        c.l_string,
        c.coords.tolist(),
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
    )

    return list(a)


if __name__ == "__main__":
    iter = 10
    pool = mp.Pool()

    pbar = tqdm(total=iter)
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
        'Average chars between -s'
    ]]

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        results = pool.apply_async(genGen, callback=update)
        population.append([f.get() for f in results])

    pool.close()
    pool.join()
    pbar.close()

    d = (col for col in population[1:])
    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)
    file_name = os.path.join(curr_dir, 'monte_carlo.csv')
    with open(file_name, 'w') as f:
        for col in population[0]:
            f.write('%s;' % col)
        for row in population[1:]:
            for column in row:
                if isinstance(column, list):
                    for col in column:
                        f.write('%s;' % str(column))
                else:
                    f.write('%s;' % str(column))
            f.write('\n')
