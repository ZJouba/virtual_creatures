from CreatureTools_n import Creature
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from datetime import datetime
import sys
import time

# from scipy.interpolate import interp1d
#
# from scipy import interpolate
# from tqdm import tqdm


def genGen():
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

    iter = 10000
    population = []

    with mp.Pool() as pool:

        for i in range(iter):
            progress(i, iter, status='Doing job')

            np.random.seed()

            results = pool.apply_async(genGen)
            population.append(results.get())

    pool.join()

    sys.stdout.write('Done! Writing to CSV')
    sys.stdout.flush()

    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)

    now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
    file_name = os.path.join(
        curr_dir, 'monte_carlo ' + now + '_.csv')

    population.to_csv(file_name, index=None, header=True,
                      chunksize=10000)
