from CreatureTools_n import Creature
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d


def genGen():
    choices = [
        'F',
        '+',
        '-',
    ]

    proba1 = np.random.uniform(0, 1)
    proba2 = 1 - proba1

    rule1 = ''.join([np.random.choice(choices) for _ in range(5)]) + 'X'
    rule2 = ''.join([np.random.choice(choices) for _ in range(5)]) + 'X'

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
    timings = []
    iterations = []

    iter = 100000

    with mp.Pool() as pool:
        pbar = tqdm(total=iter, smoothing=0.5)
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

    pool.join()
    pbar.close()

    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)

    now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
    file_name = os.path.join(
        curr_dir, 'CSVs\\monte_carlo ' + now + '_.csv')

    population.to_csv(file_name, index=None, header=True,
                      chunksize=10000)
