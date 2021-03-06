import logging
import os
import random
import sys
import time
from itertools import product
from math import atan2, cos, degrees, floor, pi, sin, sqrt

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grid_strategy import strategies
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy.spatial import KDTree
from shapely.affinity import scale
from shapely.geometry import LinearRing, LineString, Polygon

from Tools.Classes import Actuator, Limb
from Tools.Gen_Tools import delete_lines, overlay_images

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

def zdist(a,b):
        return sum(np.sqrt(((a-b)**2).sum(axis=0)))


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


def plot_limb(limbs, objt=None, curve=None):
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

        end = np.array((limb.XY[0, -num_points:],
                        limb.XY[1, -num_points:]))
        segs = len(limb.orient)

        coordinates = np.copy(limb.XY)

        # ax.set_title("Soft actuator\n" + "Number of segments: {}".format(segs))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.plot([0, 0], [-2, 2], color='black')
        ax.xaxis.set_major_locator(MultipleLocator(1))

        normal = np.zeros((len(vec)+1))
        ''' PLOT UNACTUATED LINE '''
        # ax.plot(normal, color='grey',
        #         label="Initial pressure (P=P" + r'$_i$' + ")")
        """------ACTUATED-------"""
        ax.plot(coordinates[0, :], coordinates[1, :], color='red',
                label="Final pressure (P=P" + r'$_f$' + ")", alpha=0.6)

        if isinstance(objt, np.ndarray):
            ax.plot(objt[0], objt[1], color='black')
        elif isinstance(objt, Polygon):
            ax.plot(objt.exterior.xy[0], objt.exterior.xy[1], color='black')

        ax.plot(curve[0], curve[1], color='blue', alpha=0.6)
        ''' VISUALISATION OF CURVE FIT '''
        if zdist(end, curve) > zdist(end, curve[::-1]):
            ax.plot([end[0], curve[0][::-1]],
                    [end[1], curve[1][::-1]], 
                    linewidth = 2,
                    color='green')
        else:   
            ax.plot([end[0], curve[0]], [end[1], curve[1]],
                    linewidth=2,
                    color='green')

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


def make_object(shape):
    length = parameters.get('Gripping').get('Length')
    width = parameters.get('Gripping').get('Width')
    start = parameters.get('Gripping').get('Location')
    X0 = start[0]
    Y0 = start[1]

    if shape == 'Square':
        obj = Polygon([
            (X0, Y0),
            (X0, Y0 + length),
            (X0 + width, Y0 + length),
            (X0 + width, Y0)
        ])
        return obj
    elif shape == 'Triangle':
        obj = Polygon([
            (X0, Y0),
            (X0 + (0.5*width), Y0 + length),
            (X0 + width, Y0),
        ])
        return obj
    elif shape == 'Circle':
        X = [(X0+length/2)+(cos(2*pi/100*x)*length/2) for x in range(0, 101)]
        Y = [(Y0+length/2)+(sin(2*pi/100*x)*length/2) for x in range(0, 101)]
        return np.array((X, Y))


def selection(results, popsize, choices):

    fitnesses = [item[3] for item in results]
    probabilities = (list(map(lambda x: 1/x, fitnesses)))
    total = sum(probabilities)
    probabilities = list(map(lambda x: x/total, probabilities))
    indices = list(range(popsize))

    elite = 5
    randomized = int(0.1*popsize)
    mutation = int(0.01*popsize)

    avg_len = 0

    new = []
    def getValues(x): return list(x.get('X').values())
    rules = list(map(getValues, np.array(results)[:, 2]))
    rule_array = np.array(rules)

    for i in range(elite):
        new.append(rules[0])

    for _ in range(randomized):
        l = np.random.randint(1, parameters.get('Rule length')+1)
        avg_len += l
        rule_a = ''.join(np.random.choice(choices, l))
        rule_b = ''.join(np.random.choice(choices, l))
        new.append([rule_a, rule_b])

    for _ in range(mutation):
        ind = np.random.choice(
            indices,
            p=probabilities
        )
        rule = rules[ind]
        a = list(rule[0])
        b = list(rule[1])
        if len(a) == 0 or len(b) == 0:
            new.append(['', ''])
            continue
        a_i = int(np.random.randint(0, len(a)))
        b_i = int(np.random.randint(0, len(b)))
        a[a_i] = np.random.choice(choices)
        b[b_i] = np.random.choice(choices)
        new.append([''.join(a), ''.join(b)])

    while len(new) < popsize:
        indA = np.random.choice(
            indices,
            p=probabilities
        )
        indB = np.random.choice(
            indices,
            p=probabilities
        )
        rule_a = rules[indA]
        rule_b = rules[indB]
        temp_new = []
        for i in range(len(rule)):
            r_1 = list(rule_a[i])
            r_2 = list(rule_b[i])
            n_r = r_1[:int(len(r_1)/2)] + r_2[int(len(r_1)/2):]
            temp_new += [''.join(n_r)]
        new.append(temp_new)

    return new


def gripper_GA(choices_dict):
    """
    Parameters:
        pop_size            --  Number of designs per generation
        iteration           --  Iteration counter (don't alter)
        max_iters           --  Maximum iterations before termination
        counter             --  Counter (don't alter)
        stop                --  Termination check (don't alter)
        recurs              --  Number of allowed recursions of the L-string
        dur                 --  Parameter for runtime calculations (don't alter)
        best                --  Ranking parameter (don't alter)
        patience            --  Number of grace iterations to wait for a better design

    """
    global num_points

    choices_list = ['X'] + list(choices_dict.keys())

    pop_size = 250
    iteration = 1
    max_iters = 5000
    counter = 0
    stop = False
    recurs = 5
    dur = 0
    best = np.inf
    patience = 2

    results = []
    obj = make_object(parameters.get('Gripping').get('Object'))

    if isinstance(obj, Polygon):
        curve = obj.buffer(20, join_style=1)
        size = len(curve.exterior.xy[0])
        q1 = int(0.35*size)

        data = np.array((
            curve.exterior.xy[0][q1:-5],
            curve.exterior.xy[1][q1:-5],
        ))

        rng = (size - 5) - q1
        per = rng/size
        t_length = curve.length * per

        num_points = floor(t_length/15)

        idx = np.round(np.linspace(
            0, data.shape[1] - 1, num_points)).astype(int)

        curve_plot = data
        curve_fit = data[:, idx]

    else:
        centr_x = parameters.get('Gripping').get('Location')[
            0] + (0.5*parameters.get('Gripping').get('Length'))  # + 5
        centr_y = parameters.get('Gripping').get('Location')[
            1] + (0.5*parameters.get('Gripping').get('Length'))  # + 5
        radius = (parameters.get('Gripping').get('Length')*1.2)/2 + 25
        obj_angle = atan2(centr_y, centr_x)
        start_theta = 6.28319 + 1.5708 + obj_angle
        end_theta = 6.28319 - obj_angle

        t_length = radius * pi
        num_points = floor(t_length/15)

        theta = np.linspace(start_theta, end_theta, num_points)

        x_circ = centr_x + (radius * np.cos(theta))
        y_circ = centr_y + (radius * np.sin(theta))

        curve_fit = np.array((x_circ, y_circ))
        curve_plot = curve_fit

    for _ in range(pop_size):
        rule_a = ''.join(np.random.choice(choices_list, parameters.get('Rule length')))
        rule_b = ''.join(np.random.choice(choices_list, parameters.get('Rule length')))

        rules = {'X': {1: rule_a, 2: rule_b}}

        actuator = Actuator()

        actuator.generate_string(rules, recurs)

        actuator.generate_coordinates(choices_dict)

        curve_act = np.array(
            (actuator.XY[0, -num_points:], actuator.XY[1, -num_points:]))

        if actuator.XY.shape[1] < num_points:
            fit = 999
        else:
            if any((curve_act-curve_fit).sum(axis=1)) < 0:
                penalty = 5
            else:
                penalty = 1

            if zdist(curve_act,curve_fit) > zdist(curve_act, curve_fit[::-1]):
                fit = (1/num_points)*zdist(curve_act,
                                           curve_fit[::-1]) * penalty
            else:
                fit = (1/num_points)*zdist(curve_act, curve_fit) * penalty

        results.append([
            actuator.XY,
            actuator.vector,
            rules,
            fit,
            recurs,
        ])

    results.sort(key=lambda x: x[3], reverse=False)

    recursions = results[0][4]

    next_gen = selection(results, pop_size, choices_list)
    rl = len(max(next_gen[0], key=len))

    while not stop:
        s_time = time.time()
        iteration += 1

        print('\nIteration:\t{}'.format(iteration))
        print('\nTop value:\t{}'.format(best))
        print('\nRecursions:\t{}'.format(recursions))
        print('\nRule length:\t{}'.format(rl))

        results = []
        for ea in next_gen:
            new_rule = {'X': {1: ea[0], 2: ea[1]}}

            actuator = Actuator()

            recurs = np.random.randint(1, 5)

            actuator.generate_string(new_rule, recurs)

            actuator.generate_coordinates(choices_dict)

            to_tuple_actuator = [(x, y)
                                 for x, y in zip(actuator.XY[0], actuator.XY[1])]

            if not len(to_tuple_actuator) < 2:
                if not isinstance(obj, Polygon):
                    to_tuple_object = [(x, y) for x, y in zip(obj[0], obj[1])]
                    object_ring = LinearRing(to_tuple_object)
                    inter_obj = scale(object_ring, 1.3, 1.3)
                else:
                    inter_obj = scale(obj, 1.3, 1.3)

                line = LineString(to_tuple_actuator)

                if line.intersects(inter_obj):
                    fit = 999
                    results.append([
                        actuator.XY,
                        actuator.vector,
                        new_rule,
                        fit,
                        recurs
                    ])
                else:
                    curve_act = np.array(
                        (actuator.XY[0, -num_points:], actuator.XY[1, -num_points:]))

                    if actuator.XY.shape[1] < num_points:
                        fit = 999
                    else:
                        if any((curve_act-curve_fit).sum(axis=1)) < 0:
                            penalty = 5
                        else:
                            penalty = 1

                        if zdist(curve_act, curve_fit) > zdist(curve_act, curve_fit[::-1]):
                            fit = (1/num_points)*zdist(curve_act,
                                                    curve_fit[::-1]) * penalty
                        else:
                            fit = (1/num_points)*zdist(curve_act, curve_fit) * penalty

                    results.append([
                        actuator.XY,
                        actuator.vector,
                        new_rule,
                        fit,
                        recurs
                    ])
            else:
                results.append([
                    None,
                    None,
                    {'X': {1: '', 2: ''}},
                    999,
                    recurs
                ])
        e_time = time.time()

        dur += (e_time - s_time)/pop_size

        results.sort(key=lambda x: x[3], reverse=False)
        recursions = results[0][4]
        rl = len(max(next_gen[0], key=len))

        if iteration > max_iters:
            stop = True
        if best <= results[0][3]:
            # if best < 5:
            counter += 1
        else:
            best = results[0][3]
            counter = 0
        if counter > patience:
            stop = True

        next_gen = selection(results, pop_size, choices_list)

        delete_lines(n=8)

    avg_time = dur/iteration
    print("\n" + 150*"-")
    print("\nSTOPPING CRITERIA REACHED\n")
    print("Number of iterations:\t{}\n".format(iteration))
    print("Number of designs:\t{}\n".format(iteration * pop_size))
    print("Average duration per individual:\t{:.5f} s\n".format(avg_time))
    print("Best value:\t{} \n".format(best))
    print("Rules:\t{}\n".format(list(results[0][2].get('X').values())))
    print("Number of recursions:\t{}".format(results[0][4]))
    print("\n" + 150*"-")
    plot_limb(results[0][1], obj, curve_plot)


if __name__ == '__main__':
    """Genetic Algorithm for Soft Gripping Actuator 

    Parameters:
        Rule length         --  Number of characters allowed in the L-system rule
        Gripping            --  Settings for the gripping actuator
                                Object      --  Specifies form of object used for gripping
                                Length      --  Physical length of object
                                Width       --  Physical width of object
                                Location    --  Location (x, y) of object from origin 
        
    
    Settings:
        Plot soup           --  Show plot of developed unique sub-units
        
    """

    global settings, generated_choices

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    settings = {
        'Plot soup': False,
    }

    generated_choices = primordialSoup()

    parameters = {
        'Rule length': 6,
        'Gripping': {
            'Object': 'Circle',  # 'Square', 'Triangle', 'Circle'
            'Length': 50, # 4, 8, 12
            'Width': 50, # 4, 8, 12
            'Location': (100, 100), # 5, 10, 15
        }
    }

    gripper_GA(generated_choices)
