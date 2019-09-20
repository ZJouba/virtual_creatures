import numpy as np
from itertools import product


def primordialSoup():
    grams = []
    for i in range(2, 6):
        grams.append(list(product(['TOP', 'BOTTOM'], repeat=i)))

    print()


if __name__ == '__main__':

    primordialSoup()
