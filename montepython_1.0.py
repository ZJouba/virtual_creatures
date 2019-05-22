from CreatureTools import Creature
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


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

    return Creature(params).coords


if __name__ == "__main__":

    iter = 1000
    with Pool(4) as p:
        results = list(tqdm([p.apply_async(genGen)], total=iter))
        # population.append([f.get() for f in results])

print()
