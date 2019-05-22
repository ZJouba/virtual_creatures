from CreatureTools import Creature
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os
import pandas as pd
from datetime import datetime


def genGen():
    choices = [
        'F',
        '+',
        '-',
    ]

    proba1 = np.random.uniform(0, 1)
    proba2 = 1 - proba1

    rule1 = ''.join([np.random.choice(choices) for _ in range(5)])
    rule2 = ''.join([np.random.choice(choices) for _ in range(5)])

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
    iter = 1000
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
        population.append(results.get())

    pool.close()
    pool.join()
    pbar.close()

    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)

    now = datetime.utcnow().strftime('%H.%M-%d.%m.%y')
    file_name = os.path.join(curr_dir, 'monte_carlo_' + now + '_.csv')

    population.to_csv(file_name, index=None, header=True)
