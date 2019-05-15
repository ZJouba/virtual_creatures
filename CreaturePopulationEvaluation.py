from Creatures import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool

import time

def get_individual(gram_length):
    sys = Worm(gram_length)
    sys.build_point_list()
    sys.expose_to_environment()
    return sys.creature_fitness, sys.l_string

def scan_population(num_individuals, gram_length):
    start = time.time()
    with Pool(5) as p:
        results = p.map(get_individual, [gram_length] * num_individuals)
    end = time.time()
    print(end - start)
    fitness, phenotypes = zip(*results)
    plt.hist(fitness, bins=50)
    return fitness, phenotypes


if __name__ == '__main__':
    scan_population(1000, 500)