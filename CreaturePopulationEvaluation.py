from Creatures import *
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from multiprocessing import Pool

import time

import pandas as pd


def monte_carlo(creature, num_individuals, pool_size=8):
    start = time.time()
    with Pool(pool_size) as p:
        # results = list(tqdm.tqdm(p.imap(get_individual,
        #                           [gram_length] * num_individuals),
        #                    total=num_individuals))
        results = list(tqdm.tqdm(p.imap(creature.get_params,
                                        [1] * num_individuals),
                                 total=num_individuals))
    end = time.time()
    print("Time to simulate {}seconds".format(end - start))

    return results


def build_df(data):
    df = pd.DataFrame(data)
    df.columns = ["fitness",
                  "points",
                  "creature_length",
                  "creature_feed_zone",
                  "gram"]
    return df


def plot_distribution(df, field):
    sns.distplot(df[field], kde=False)


if __name__ == '__main__':
    population = 100000
    gram_length = 100

    params = {"num_char": gram_length,
              "variables": "X",
              "constants": "F+-",
              "axioms": "FX",
              "rules": {"X": {"options": ["FX", "+X", "-X"],
                              "probabilities": [0.4, 0.3, 0.3]}},
              "point": np.array([0, 0]),
              "vector": np.array([0, 1]),
              "length": 1.0,
              "angle": 25,
              "feed_radius": 0.5,
              "len_scale_factor": 1,
              "angle_inc": 0}
    sys = Worm(params)

    res = monte_carlo(sys, population, gram_length)
    df = build_df(res)
    plot_distribution(df, "fitness")
    plot_distribution(df, "creature_length")

    sns.scatterplot(x="creature_length", y="fitness", data=df)