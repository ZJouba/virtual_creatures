from CreatureTools import Creature
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp


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
    # def parallel_monte_carlo(iter):
    pool = mp.Pool(4)

    pbar = tqdm(total=iter)
    population = []

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        results = [pool.apply_async(genGen, callback=update)]
        population.append([f.get() for f in results])

    pool.close()
    pool.join()

    print(population[0])
    # return res

    # final_results = parallel_monte_carlo(100000)

    # pbar = tqdm(total=generations, position=2, unit='generation',
    #             ascii=True, dynamic_ncols=True, smoothing=0.5)

    # while generation < (generations + 1):
    #     params = {
    #         'num_char': 100,
    #         'variables': 'X',
    #         'constants': 'F+-',
    #         'axiom': 'FX',
    #         'rules': {
    #             'X': {
    #                 'options': ['+FX', '-FX'],
    #                 'probabilities': [0.5, 0.5]
    #             }
    #         },
    #         'point': np.array([0, 0]),
    #         'vector': np.array([0, 1]),
    #         'length': 1.0,
    #         'angle': 25  # random
    #     }

    #     generation += 1

    #     population.append(Creature(params).coords)

    #     pbar.update()

print()
