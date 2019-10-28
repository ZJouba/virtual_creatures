from CreatureTools_n import Creature
from Tools.Classes import Environment
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

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from descartes import PolygonPatch


def genGen(params, listed=False):
    proba1 = np.random.uniform(0, 1)
    proba2 = 1 - proba1

    rule1 = ''.join([np.random.choice(params.get('choices'))
                     for _ in range(params.get('rule_length'))])
    rule2 = ''.join([np.random.choice(params.get('choices'))
                     for _ in range(params.get('rule_length'))])

    params['rules'] = {'X': {1: rule1, 2: rule2}}

    params['angle'] = np.random.randint(0, 90)  # random

    c = Creature(params)
    try:
        c = Creature(params)
    except:
        pass

    if listed:
        return list(c.__dict__.keys())
    else:
        return list(c.__dict__.values())


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


if __name__ == "__main__":

    params = {
        'iterations': 100000,
        'recurs': 5,
        'variables': 'X',
        'constants': 'F+-_',
        'axiom': 'FX',
        'length': 1.0,
        'rule_length': 5,
        'fitness_metric': 'Area',
        'shape': 'square',  # 'circle' 'square' 'rainbow' 'triangle' 'patches'
        'richness': 'common',  # 'scarce' 'common' 'abundant'
        'scale': 'small',  # 'small' 'medium' 'large'
    }

    params['choices'] = list(params.get(
        'variables') + params.get('constants'))

    # fig, ax = plt.subplots()

    # env = Environment(params)

    # params['env'] = env

    # p = PolygonPatch(env.patches[0])

    # ax.add_patch(p)

    # plt.show()

    init_creature = genGen(params, listed=True)

    population = [init_creature]

    # for _ in range(5):
    #     genGen(params)

    with mp.Pool(mp.cpu_count()-2) as pool:
        np.random.seed()
        results = list(
            tqdm.tqdm(pool.imap(genGen, repeat(params, params.get('iterations'))), total=params.get('iterations')))
        population = population + results

    pool.join()

    sys.stdout.write('Done! Writing to CSV')
    sys.stdout.flush()

    population = pd.DataFrame(population[1:], columns=population[0])

    curr_dir = os.path.dirname(__file__)

    now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
    file_name = os.path.join(
        curr_dir, 'CSVs/branch_monte_carlo ' + now + '.p')

    pickle.dump(population, open(file_name, 'wb'))
