import multiprocessing as mp 
import numpy as np
import tqdm
from itertools import repeat
import os
from datetime import datetime
import pickle
import pandas as pd

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

    c = Creature(params)
    a = (
        c.l_string,
        c.area,
        c.rules,
        c.choices,
        c.ratio,
    )

    return list(a)

def firstRun(iter):
    population = []
    population = [[
        'L-string',
        'Area',
        'Rules',
        'Choice vector',
        'Ratios',
    ]]

    with mp.Pool(6) as pool:
        result = list(tqdm.tqdm(pool.imap(genPop, repeat(params, iter)), total=iter))
        population = population + result  

    return population

if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)

    params = {
            'chars': 500,
            'recurs': 100,
            'variables': 'X',
            'constants': 'F+-',
            'axiom': 'FX',
            'length': 1.0,
            'angle': 25,
            'prune': True,
            'pairwise': False,
        }

    population = firstRun(50000)
    population = pd.DataFrame(population[1:], columns=population[0])

    now = datetime.utcnow().strftime('%b %d, %Y @ %H.%M')
    file_name = os.path.join(
            curr_dir, 'CSVs/mini_monte_data ' + now + '.p')
        
    pickle.dump(population, open(file_name, 'wb'))