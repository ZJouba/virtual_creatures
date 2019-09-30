import os
import random
from itertools import product
from math import degrees, pi, cos, sin

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from grid_strategy import strategies
from tabulate import tabulate
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, LinearRing

from Tools.Classes import Limb, Actuator
from Tools.Gen_Tools import overlay_images, delete_lines


def evaluate(vector):
    l = Limb()

    l.build(vector)

    num_segments = len(vector)
    tip_x = round(l.XY[0][-1], 3)
    tip_y = round(l.XY[1][-1], 3)
    curvature = degrees(sum(l.curvature))

    eval_results = [
        vector,
        len(vector),
        tip_x,
        tip_y,
        curvature,
        tip_x / num_segments,
        tip_y / num_segments,
        curvature / (num_segments**2),
    ]

    return eval_results


def plot(subunits):
    """Plots actuators

    Arguments:
        subunits {list} -- List of subunit ORIENTATION VECTORS
    """

    num_plots = len(subunits)

    specs = strategies.SquareStrategy('center').get_grid(num_plots)

    fig = plt.figure(1)

    for vec, sub in zip(subunits, specs):
        ax = fig.add_subplot(sub)

        l = Limb()
        l.build(vec)

        ax.plot([0, 0], [-2, 2], color='black')
        ax.plot(l.XY[0], l.XY[1], color='red')

        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(1.5, 1.5)
        ax.autoscale(False)
        overlay_images(ax, l)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_limb(limbs, objt=None):
    for i in range(len(limbs[0])):
        if isinstance(limbs[0][i], list):
            vec_ind = i

    limbs = np.array(limbs)

    if len(limbs.shape) > 1:
        num_plots = len(limbs)
        limbs = limbs[:, vec_ind]
    else:
        num_plots = len(limbs.shape)
        limbs = [limbs]

    specs = strategies.SquareStrategy('center').get_grid(num_plots)

    fig = plt.figure(1, constrained_layout=False)
    fig.canvas.set_window_title('Actuator')

    for vec, sub in zip(limbs, specs):
        ax = fig.add_subplot(sub)

        limb = Limb()
        limb.build(vec)

        segs = len(limb.orient)

        coordinates = np.copy(limb.XY)

        ax.set_title("Soft actuator\n" + "Number of segments: {}".format(segs))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.plot([0, 0], [-2, 2], color='black')
        ax.xaxis.set_major_locator(MultipleLocator(1))

        normal = np.zeros((len(vec)+1))
        ax.plot(normal, color='grey',
                label="Initial pressure (P=P" + r'$_i$' + ")")
        """------ACTUATED-------"""
        ax.plot(coordinates[0, :], coordinates[1, :], color='red',
                label="Final pressure (P=P" + r'$_f$' + ")")

        if isinstance(objt, np.ndarray):
            ax.plot(objt[0], objt[1], color='black')

        ax.margins(0.5, 0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale(False)
        overlay_images(ax, limb)

    plt.tight_layout()
    plt.show()


def primordialSoup():

    grams = []
    for i in range(2, 5):
        grams.append(list(product(['TOP', 'BOTTOM'], repeat=i)))

    grams = [item for sublist in grams for item in sublist]
    grams = [evaluate(gram) for gram in grams]

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

    frame = pd.DataFrame(grams, columns=index)

    frame = frame.loc[frame['INDIVIDUAL'].abs(
    ).drop_duplicates().index]

    frame.drop_duplicates(subset=[('INDIVIDUAL', 'CURVATURE')], inplace=True)
    frame.drop_duplicates(subset=[
        ('INDIVIDUAL', 'TIP X'),
        ('INDIVIDUAL', 'TIP Y')
    ], inplace=True)

    frame.reset_index(drop=True, inplace=True)

    A = frame['INDIVIDUAL']
    xy = A[['TIP X', 'TIP Y']].values
    mask = [item[1] for item in list(KDTree(xy).query_pairs(0.1))]
    mask = list(dict.fromkeys(mask))

    frame.drop(frame.index[mask], inplace=True)
    frame.reset_index(drop=True, inplace=True)

    plotable = [list(elem) for elem in list(frame['VECTOR'])]
    if settings.get('Plot soup'):
        plot(plotable)

    choice_dict = {}
    for i, unit in enumerate(plotable):
        char = chr(i+97)
        choice_dict[char] = unit

    return choice_dict


def make_object():
    length = parameters.get('Gripping').get('Length')
    width = parameters.get('Gripping').get('Width')
    start = parameters.get('Gripping').get('Location')

    if parameters.get('Gripping').get('Object') == 'Square':
        X = 2*[start[0]] + 2*[length + start[0]] + [start[0]]
        Y = [start[1]] + 2*[length + start[1]] + 2*[start[1]]
        return np.array((X, Y))
    elif parameters.get('Gripping').get('Object') == 'Triangle':
        X = [start[0]] + [0.5*length + start[0]] + \
            [length + start[0]] + [start[0]]
        Y = [start[1]] + [width + start[1]] + 2*[start[1]]
        return np.array((X, Y))
    elif parameters.get('Gripping').get('Object') == 'Circle':
        X = [start[0]+(cos(2*pi/100*x)*length) for x in range(0, 101)]
        Y = [start[1]+(sin(2*pi/100*x)*length) for x in range(0, 101)]
        return np.array((X, Y))


def selection(results, popsize, choices):

    elite = 1
    randomized = int(0.3*popsize)
    mutation = int(0.4*popsize)

    new = []
    rules = results['Rules'].apply(lambda x: list(x.get('X').values()))
    rules = rules.tolist()

    for i in range(elite):
        new.append(rules[0])

    for _ in range(randomized):
        l = np.random.randint(1, 10)
        rule_a = ''.join(np.random.choice(choices, 5))
        rule_b = ''.join(np.random.choice(choices, 5))
        new.append([rule_a, rule_b])

    for _ in range(mutation):
        rule = random.choice(rules)
        a = list(rule[0])
        b = list(rule[1])
        a_i = int(np.random.randint(0, len(a)))
        b_i = int(np.random.randint(0, len(a)))
        a[a_i] = np.random.choice(choices)
        b[b_i] = np.random.choice(choices)
        new.append([''.join(a), ''.join(b)])

    while len(new) < popsize:
        rule_a = random.choice(rules)
        rule_b = random.choice(rules)
        temp_new = []
        for i in range(len(rule)):
            r_1 = list(rule_a[i])
            r_2 = list(rule_b[i])
            n_r = r_1[:int(len(r_1)/2)] + r_2[int(len(r_1)/2):]
            temp_new += [''.join(n_r)]
        new.append(temp_new)
    return new


def gripper_GA(choices_dict):

    choices_list = ['X'] + list(choices_dict.keys())

    pop_size = 100
    iteration = 1
    max_iters = 200
    stop = False
    recurs = 5

    results = []
    obj = make_object()

    for _ in range(pop_size):
        rule_a = ''.join(np.random.choice(choices_list, 5))
        rule_b = ''.join(np.random.choice(choices_list, 5))

        rules = {'X': {1: rule_a, 2: rule_b}}

        actuator = Actuator()

        actuator.generate_string(rules, recurs)

        actuator.generate_coordinates(choices_dict)

        # plot_limb(actuator.vector, obj)

        curve = actuator.XY[:, -3:]

        fit = np.linalg.norm(cdist(curve, obj[:, -3:], 'sqeuclidean'))

        results.append([
            actuator.XY,
            actuator.vector,
            rules,
            fit,
        ])

    results = pd.DataFrame(
        results, columns=['XY', 'Vector', 'Rules', 'Fitness'])

    results.sort_values('Fitness', ascending=True, inplace=True)

    next_gen = selection(results, pop_size, choices_list)

    while not stop:
        iteration += 1
        print('\nIteration:\t{}'.format(iteration))
        # print('\nTop value:\t{}'.format(float(best)))

        delete_lines(n=2)

        for i, ea in enumerate(next_gen):
            new_rule = {'X': {1: ea[0], 2: ea[1]}}

            actuator = Actuator()

            actuator.generate_string(new_rule, recurs)

            actuator.generate_coordinates(choices_dict)

            to_tuple_actuator = [(x, y)
                                 for x, y in zip(actuator.XY[0], actuator.XY[1])]

            if not len(to_tuple_actuator) < 2:
                to_tuple_object = [(x, y) for x, y in zip(obj[0], obj[1])]
                object_ring = LinearRing(to_tuple_object)

                lines = []
                lines.append(LineString(to_tuple_actuator))
                try:
                    lines.append(lines[0].parallel_offset(1.9, side='left'))
                except:
                    pass
                try:
                    lines.append(lines[0].parallel_offset(1.9, side='right'))
                except:
                    pass

                for each in lines:
                    if object_ring.intersects(each):
                        fit = np.inf
                        break
                    else:
                        curve = actuator.XY[:, -3:]
                        fit = np.linalg.norm(
                            cdist(curve, obj[:, -3:], 'sqeuclidean'))

                results.iloc[i] = (
                    actuator.XY,
                    actuator.vector,
                    rules,
                    fit,
                )

            else:
                results.iloc[i] = (
                    None,
                    None,
                    {'X': {1: '', 2: ''}},
                    np.inf,
                )

        results.sort_values('Fitness', ascending=True, inplace=True)

        next_gen = selection(results, pop_size, choices_list)

        if iteration > max_iters:
            stop = True

    plot_limb(results['Vector'].iloc[0], obj)


if __name__ == '__main__':
    global settings

    settings = {
        'Plot soup': False,
    }

    generated_choices = primordialSoup()

    parameters = {
        'Rule length': 5,
        'Gripping': {
            'Object': 'Square',  # 'Square', 'Triangle', 'Circle'
            'Length': 10,
            'Width': 10,
            'Location': (10, 10),
        }
    }

    gripper_GA(generated_choices)
