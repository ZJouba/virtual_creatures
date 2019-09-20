import numpy as np
from itertools import product
import pandas as pd
from tabulate import tabulate
from Tools.Classes import Limb


def evaluate(vector):
    l = Limb()

    l.build(vector)

    num_segments = len(vector)
    tip_x = l.XY[0][-1]
    tip_y = l.XY[1][-1]
    curvature = l.curvature

    eval_results = [
        vector,
        len(vector),
        [
            tip_x,
            tip_y,
            curvature,
        ],
        [
            tip_x / num_segments,
            tip_y / num_segments,
            curvature / num_segments,
        ],
    ]

    return eval_results


def primordialSoup():

    grams = []
    for i in range(2, 6):
        grams.append(list(product(['TOP', 'BOTTOM'], repeat=i)))

    frameHeaders = [
        ['VECTOR',
         'SEGMENTS',
         'OVERALL',
         'OVERALL',
         'OVERALL',
         'INDIVIDUAL',
         'INDIVIDUAL',
         'INDIVIDUAL'],
        ["",
         "",
         'TIP X',
         'TIP Y',
         'CURVATURE',
         'TIP X',
         'TIP Y',
         'CURVATURE'],
    ]

    index = pd.MultiIndex.from_arrays(frameHeaders)

    frames = {}
    for i, gram in enumerate(grams, 2):
        frames[i] = pd.DataFrame(evaluate(gram), columns=index)

    print()


if __name__ == '__main__':

    primordialSoup()
