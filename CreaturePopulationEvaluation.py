from Creatures import *
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from multiprocessing import Pool

import time

import pandas as pd


def get_individual(gram_length):
    params = {"num_char": gram_length,
              "variables": "X",
              "constants": "F+-",
              "axioms": "FX",
              "rules": {"X": {"options": ["FX", "-X", "-X"],
                              "probabilities": [0.4, 0.3, 0.3]}},
              "point": np.array([0, 0]),
              "vector": np.array([0, 1]),
              "length": 1.0,
              "angle": 25}
    sys = Worm(params)
    sys.build_point_list()
    sys.expose_to_environment()
    return sys
    # return sys.creature_fitness, sys.l_string


def scan_population(num_individuals, gram_length=100, pool_size=8):
    start = time.time()
    with Pool(pool_size) as p:
        results = list(tqdm.tqdm(p.imap(get_individual,
                                  [gram_length] * num_individuals),
                           total=num_individuals))
    end = time.time()
    print("Time to simulate {}seconds".format(end - start))

    return results


def build_df(res):
    data = [(sys.l_string,
             sys.creature_fitness,
             sys.creature_length,
             sys.creature_feed_zone) for sys in res]

    # df = pd.DataFrame(data, index=["lstring", "fitness"])
    df = pd.DataFrame(data)
    df.columns = ["lstring", "fitness", "length", "area"]
    return df


def plot_distribution(df, field):
    # sns.distplot(df[field], label=field, hist=True, norm_hist=True)
    # df[field].hist()
    sns.distplot(df[field], kde=False)


if __name__ == '__main__':
    population = 1000
    gram_length = 100
    res = scan_population(population, gram_length)
    df = build_df(res)
    plot_distribution(df, "fitness")
    # plot_distribution(df, "length")