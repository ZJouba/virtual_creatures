import numpy as np

if __name__ == '__main__':
    """
    'Extend',
    'Curl clockwise',
    'Curl anticlockwise',
    """

    choices = [
        'E',
        'CC',
        'CA',
    ]

    generation = []

    while len(generation) <= 500:
        genotype = [np.random.choice(choices) for _ in range(10)]
        generation.append(genotype)
