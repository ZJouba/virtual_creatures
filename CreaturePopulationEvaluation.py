from Creatures import *
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool

import time

import pandas as pd


def get_individual(gram_length):
    params = {"num_char": 100,
              "variables": "X",
              "constants": "F+-",
              "axioms": "FX",
              "rules": {"X": {"options": ["+FX", "-FX"],
                              "probabilities": [0.5, 0.5]}},
              "point": np.array([0, 0]),
              "vector": np.array([0, 1]),
              "length": 1.0,
              "angle": 25}
    sys = Worm(params)
    sys.build_point_list()
    sys.expose_to_environment()
    return sys
    # return sys.creature_fitness, sys.l_string


def scan_population(num_individuals, gram_length, pool_size=8):
    start = time.time()
    with Pool(pool_size) as p:
        results = list(tqdm.tqdm(p.imap(get_individual,
                                  [gram_length] * num_individuals),
                           total=num_individuals))
    end = time.time()
    print(end - start)
    # fitness, phenotypes = zip(*results)
    # plt.hist(fitness, bins=50)
    # plt.show()
    return results


def build_df(res):
    df = pd.DataFrame(zip(res.l_string,
                          red.creature_fitness))
    return df


if __name__ == '__main__':
    population = 1000
    res = scan_population(population, size)