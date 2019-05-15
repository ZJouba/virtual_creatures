from Creatures import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_individual(gram_length):
    sys = Worm(gram_length)
    sys.build_point_list()
    sys.expose_to_environment()
    return sys.creature_fitness, sys.l_string

def scan_population(num_individuals, gram_length):
    fitness = []
    string = []
    for _ in tqdm(range(num_individuals)):
        f, s = get_individual(gram_length)
        fitness.append(f)
        string.append(s)
    plt.hist(fitness, bins=50)


scan_population(1000, 500)