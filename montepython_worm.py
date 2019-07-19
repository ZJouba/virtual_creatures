from CreatureTools_n import B_Creature
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from datetime import datetime
import sys
import time
import pickle
from itertools import repeat
import tqdm


def genGen(params):
    choices = [
        'F',
        '+',
        '-',
        'X',
    ]

    proba1 = np.random.uniform(0, 1)
    proba2 = 1 - proba1

    rule1 = ''.join([np.random.choice(choices)
                     for _ in range(5)])
    rule2 = ''.join([np.random.choice(choices)
                     for _ in range(5)])

    params['rules'] = {
        'X': {
            'options': [
                rule1,
                rule2,
            ],
            'probabilities': [proba1, proba2]
        }
    }
    params['angle'] = np.random.randint(0, 90)  # random

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
    sys.stdout.flush()


if __name__ == "__main__":

    iter = 100000
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
        'Longest F sequence',
        'Longest + sequence',
        'Longest - sequence',
        'Average chars between Fs',
        'Average chars between +s',
        'Average chars between -s',
        'Angle',
        'Rules',
    ]]

    params = {
        'num_char': 5,
        'variables': 'X',
        'constants': 'F+-',
        'axiom': 'FX',
        'point': np.array([0, 0]),
        'vector': np.array([0, 1]),
        'length': 1.0,
    }

    with mp.Pool() as pool:
        np.random.seed()
        # results = list(pool.imap(genGen, repeat(params, iter)))
        results = list(
            tqdm.tqdm(pool.imap(genGen, repeat(params, iter)), total=iter))
        population = population + results

    pool.join()

    sys.stdout.write('Done! Writing to CSV')
    sys.stdout.flush()

    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)

    now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
    file_name = os.path.join(
        curr_dir, 'CSVs/monte_carlo ' + now + '.p')

    pickle.dump(population, open(file_name, 'wb'))
