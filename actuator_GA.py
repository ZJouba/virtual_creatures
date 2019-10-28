import itertools
import multiprocessing as mp
import os
import sys
import operator
import copy
import time
from datetime import datetime
from decimal import Decimal
from math import atan2, cos, degrees, radians, sin, tan, pi


import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial as P
from matplotlib import transforms
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from scipy import ndimage
from tabulate import tabulate

from grid_strategy import strategies
from Tools.Classes import Limb

from shapely.geometry import LineString
from scipy.spatial.distance import cdist


def evaluate(orient_vector, first=False):

    global top_X, parameters

    invalid = False

    l = Limb()

    l.build(orient_vector)

    data = np.copy(l.XY)

    origin = np.array((0, 0))
    point = np.array((data[0][-1], data[1][-1]))
    D = np.linalg.norm(point-origin)
    X = data[0][-1]
    Y = data[1][-1]

    if any(value == True for value in parameters.get('Curve fitting').values()):
        m = data[0][-1]/(2*pi)
        if parameters.get('Curve fitting').get('Sin'):
            curve = np.sin(data[0]/m)
        elif parameters.get('Curve fitting').get('Cos'):
            curve = np.cos(data[0]/m)-1
        elif parameters.get('Curve fitting').get('Custom'):
            curve = []
            func = parameters.get('Curve fitting').get('Custom func')
            for ea in data[0]/m:
                x = ea
                curve.append(eval(func))

        curve_fit = cdist([data[1]], [curve], 'sqeuclidean')
    else:
        curve_fit = 0
    
    Curvature = degrees(l.curvature[-1])

    limb_res = [
        round(X, 2),
        round(Y, 2),
        round(D, 2),
        round(Curvature, 2),
        orient_vector,
        curve_fit,
    ]

    lowest = min(np.array(top_X)[:,sort_by])
    highest = max(np.array(top_X)[:,sort_by])

    if (lowest < limb_res[sort_by] < highest) or first:
        lines = []
        to_tuple = [(x, y) for x, y in zip(data[0], data[1])]
        lines.append(LineString(to_tuple))
        try:
            lines.append(lines[0].parallel_offset(1.9, side='left'))
        except:
            pass
        
        try:
            lines.append(lines[0].parallel_offset(1.9, side='right'))
        except:
            pass

        for line in lines:
            if not line.is_simple or line.is_closed:
                invalid = True

    if invalid:
        if descending:
            D = 0
            X = 0
            Y = 0

            Curvature = 0

        else:
            D = 99999
            X = 99999
            Y = 99999

            Curvature = 99999

        limb_res = [
            round(X, 2),
            round(Y, 2),
            round(D, 2),
            round(Curvature, 2),
            # l,
            orient_vector
        ]


    return limb_res


def selection(generation_data, num_segments, elite, randomised, mutation, crossover):
    global random_array

    for i in range(len(generation_data[0])):
        if isinstance(generation_data[0][i],list):
            vec_ind = i

    if isinstance(elite, float):
        elite = int(len(generation_data) * elite)
    if isinstance(randomised, float):
        randomised = int(len(generation_data) * randomised)
    elif isinstance(randomised, tuple):
        try:
            if len(random_array) == 1:
                randomised = int(len(generation_data) * random_array[-1])
            else:
                randomised, random_array = random_array[0], random_array[1:]
                randomised = int(len(generation_data) * randomised)
        except NameError:
            random_array = np.linspace(
                randomised[0], randomised[1], randomised[2])
            randomised, random_array = random_array[0], random_array[1:]
            randomised = int(len(generation_data) * randomised)

    if isinstance(mutation, float):
        mutation = int(len(generation_data) * mutation)
    if isinstance(crossover, float):
        crossover = int(len(generation_data) * crossover)

    new_generation_data = []

    for i in range(elite):
        new_generation_data.append(generation_data[i][vec_ind])

    if isinstance(num_segments, int):
        for i in range(randomised):
            new_vec = [np.random.choice(choices)
                      for _ in range(num_segments)]
            new_generation_data.append(new_vec)
    else:
        for i in range(randomised):
            top_margin = len(elite[0]) + 5
            bottom_margin = len(elite[0]) - 5
            new_vec = [np.random.choice(
                choices) for _ in range(np.random.randint(bottom_margin, top_margin))]
            new_generation_data.append(new_vec)

    for i in range(mutation):
        limb = generation_data[np.random.randint(len(generation_data))][vec_ind]
        index = np.random.randint(
            low=0, high=len(limb), size=int(len(limb)*0.2))
        for place in index:
            limb[place] = np.random.choice(choices)
        new_generation_data.append(limb)

    if crossover == 'rest':
        while len(new_generation_data) < len(generation_data):
            parentA = generation_data[np.random.randint(
                len(generation_data))][vec_ind]
            parentB = generation_data[np.random.randint(
                len(generation_data))][vec_ind]
            mid = int(len(parentA) * 0.5)
            child = parentA[:mid] + parentB[mid:]
            new_generation_data.append(child)
    else:
        for i in range(crossover):
            parentA = generation_data[np.random.randint(
                len(generation_data))][vec_ind]
            parentB = generation_data[np.random.randint(
                len(generation_data))][vec_ind]
            mid = int(len(parentA) * 0.5)
            child = parentA[:mid] + parentB[mid:]
            new_generation_data.append(child)

    return new_generation_data


def GA(parameters):
    """ ------------------------------------------------------------------------
    INITIALIZATION
    -------------------------------------------------------------------------"""
    global descending, sort_by, choices, top_X

    def delete_lines(n=1):
        for _ in range(n):
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')

    i = 0
    start = time.time()

    if settings.get('Save data'):
        current_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'CSVs'
        )
        unique_id = datetime.now().strftime('%d%m%H%M')
        file_name = 'ActuatorGA_000' + str(i) + unique_id + '.txt'
        file_path = os.path.join(
            current_directory,
            file_name
        )
        while os.path.isfile(file_path):
            i += 1
            file_name = 'ActuatorGA_000' + str(i) + unique_id + '.txt'
            file_path = os.path.join(
                current_directory,
                file_name
            )

        generation_save_directory = open(file_path, 'a')

    columns = [
        "MAX X",
        "MAX Y",
        "MAX DISTANCE",
        "TOTAL CURVATURE",
        "ORIENTATION VECTOR",
        "CURVE FIT",
    ]
    results = [[
        columns[0],
        columns[1],
        columns[2],
        columns[3],
        columns[4],
        columns[5],
    ]]

    """ ------------------------------------------------------------------------
    CHECKS
    -------------------------------------------------------------------------"""
    if settings.get('Save data'):
        generation_save_directory.write(("{}\n".format(results[0])))

    pop_size = parameters.get('Population size')

    if any(value == True for value in parameters.get('Fitness Metric').get('Metrics').values()) and \
       any(value == True for value in parameters.get('Curve fitting').values()):
        exception_string = ('Cannot optimize for fitness metric and curve fitting at the same time. Check parameters dictionary')
        raise Exception(exception_string)

    if sum(value == True for value in parameters.get('Fitness Metric').get('Metrics').values()) > 1:
        exception_string = (
            'Only one fitness metric allowed. Check parameters dictionary')
        raise Exception(exception_string)
    
    if sum(value == True for value in parameters.get('Fitness Metric').get('Metrics').values()) == 0 and \
    sum(value == True for value in parameters.get('Curve fitting').values()) == 0:
        exception_string = ('Please choose a fitness metric. Check parameters dictionary')
        raise Exception(exception_string)


    if sum(value == True for value in parameters.get('Curve fitting').values()) > 1:
        exception_string = (
            'Only one curve fit allowed. Check parameters dictionary')
        raise Exception(exception_string)

    if any(value == True for value in parameters.get('Curve fitting').values()):
        descending = False
        sort_by = 5

    else:
        fitness = parameters.get('Fitness Metric').get('Metrics')
        descending = parameters.get('Fitness Metric').get('Maximise')
        metric = [i for i, x in fitness.items() if x]
        sort_by = list(fitness.keys()).index(metric[0])
    
    num_segments = parameters.get('Number of segments').get('Type')
    if num_segments == 'Integer':
        num_segments = parameters.get('Number of segments').get('Number')
    else:
        num_segments = 0
    patience = parameters.get('Patience')
    maximum = parameters.get('Maximum iterations')
    tolerance = parameters.get('Tolerance')
    elite = parameters.get('Selection').get('Elite')
    rand = parameters.get('Selection').get('Random')
    mutation = parameters.get('Selection').get('Mutation')
    crossover = parameters.get('Selection').get('Crossover')
    choices = parameters.get('Choices')
    num_top = parameters.get('Top')

    check_set = set(str(value).lower() for value in parameters.get(
        'Selection').values())
    if not 'rest' in check_set:
        exception_string = (
            'One selection method must be set to \'rest\'. Check parameters dictionary')
        raise Exception(exception_string)

    cumm_percentage = 0
    for sel_value in parameters.get('Selection').values():
        if isinstance(sel_value, int):
            cumm_percentage += sel_value / pop_size
        elif isinstance(sel_value, tuple):
            cumm_percentage += sel_value[0]
        elif isinstance(sel_value, float):
            cumm_percentage += sel_value

    if cumm_percentage > 1:
        exception_string = (
            'Cummulative selection values cannot exceed 100%. Check parameters dictionary')
        raise Exception(exception_string)
    
    """ ------------------------------------------------------------------------
    RANDOM INITIAL POPULATION
    -------------------------------------------------------------------------"""
    top_X = [[0] * len(results[0])]*num_top

    while len(results) <= pop_size:
        if num_segments == 0:
            orientations = [np.random.choice(choices) for _ in range(np.random.randint(1, 50))]
        else:
            orientations = [np.random.choice(choices)
                            for _ in range(num_segments)]

        results.append(evaluate(orientations, True))

    new_list = results[1:]
    new_list.sort(key=lambda x: x[sort_by], reverse=descending)
    results = new_list

    indi_time = 0
    top = parameters.get('Top')
    top_X = copy.deepcopy(results[:top])
    best = top_X[0][sort_by]
    stop_criteria_counter = 0
    iterations = 0
    stop = False
    iteration = 0

    if descending:
        op = operator.ge
    else:
        op = operator.le 

    while not stop:
        indi_s_time = time.time()

        new_generation = selection(results, num_segments, elite, rand, mutation, crossover)

        results = [evaluate(ea) for ea in new_generation]

        results.sort(key=lambda x: x[sort_by], reverse=descending)

        gen_top = results[0][sort_by]

        lowest = float(min(np.array(top_X)[:,sort_by]))
        highest = float(max(np.array(top_X)[:,sort_by]))

        top_range = np.arange(lowest, highest)

        if descending:
            checker = highest
        else:
            checker = lowest

        new_place = top-1
        if (gen_top in top_range) or op(gen_top, checker):
            for place in range(top-2):
                if op(gen_top, top_X[place][sort_by]):
                    new_place = place

            for i in range(new_place, top-2):
                top_X[i+1] = top_X[i]

            top_X[new_place] = results[0]
            top_X.sort(key=lambda x: x[sort_by], reverse=descending)

        if parameters.get('Stopping criteria').get('Maximum iterations') and parameters.get('Stopping criteria').get('Tolerance'):
            iterations += 1
            if iterations > maximum:
                stop = True
            if op(best, gen_top):
                stop_criteria_counter += 1
            else:
                stop_criteria_counter = 0
            if stop_criteria_counter > patience:
                stop = True
        elif parameters.get('Stopping criteria').get('Tolerance'):
            if op(best, gen_top):
                stop_criteria_counter += 1
            else:
                stop_criteria_counter = 0

            if stop_criteria_counter > patience:
                stop = True
        elif parameters.get('Stopping criteria').get('Maximum iterations'):
            iterations += 1
            if iterations > maximum:
                stop = True

        iteration += 1

        best = top_X[0][sort_by]

        print('\nIteration:\t{}'.format(iteration))
        print('\nTop value:\t{}'.format(float(best)))

        delete_lines(n=4)

        indi_e_time = time.time()

        indi_time += (indi_e_time - indi_s_time)/pop_size

        if settings.get('Save data'):
            for line in results[1:]:
                generation_save_directory.write(("{}\n".format(line)))
    
    avg_time = indi_time/iteration
    end = time.time()
    print("\n" + 150*"-")
    if iterations > maximum:
        print("\nMAXIMUM ITERATIONS REACHED\n")
    else:
        print("\nSTOPPING CRITERIA REACHED\n")
    print("Number of iterations:\t{}\n".format(iteration))
    print("Duration:\t{:.5f} s\n".format(end - start))
    print("Average duration per individual:\t{:.5f} s".format(avg_time))
    print("\n" + 150*"-")

    if settings.get('Save data'):
        generation_save_directory.close()

    check = [evaluate(ea) for ea in np.array(top_X)[:,4]]
    check.sort(key=lambda x: x[sort_by], reverse=descending)

    if settings.get('Plot final'):
        plot_limb(check)


def plot_limb(limbs):

    def on_click(event):
        ax = event.inaxes

        if ax is None:
            return

        if event.button != 1:
            return

        if zoomed_axes[0] is None:
            zoomed_axes[0] = (ax, ax.get_position())
            ax.set_position([0.1, 0.1, 0.8, 0.8])
            ax.legend(loc='best')
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(True)

            for axis in event.canvas.figure.axes:
                if axis is not ax:
                    axis.set_visible(False)

        else:
            zoomed_axes[0][0].set_position(zoomed_axes[0][1])
            zoomed_axes[0] = None
            ax.get_legend().remove()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            for axis in event.canvas.figure.axes:
                axis.set_visible(True)

        event.canvas.draw()

    zoomed_axes = [None]

    for i in range(len(limbs[0])):
        if isinstance(limbs[0][i],list):
            vec_ind = i

    limbs = np.array(limbs)    

    if len(limbs.shape) > 1:
        num_plots = len(limbs)
        limbs = limbs[:,vec_ind]
    else:
        num_plots = len(limbs.shape)
        limbs = [limbs]

    specs = strategies.SquareStrategy('center').get_grid(num_plots)

    fig = plt.figure(1, constrained_layout=False)
    fig.canvas.set_window_title('Top ' + str(num_plots))

    for vec, sub in zip(limbs, specs):
        ax = fig.add_subplot(sub)

        limb = Limb()
        limb.build(vec)

        segs = len(limb.orient)

        points = np.copy(limb.XY)
        rots = limb.curvature

        ax.set_title("Soft actuator\n" + "Number of segments: {}".format(segs))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.plot([0, 0], [-2, 2], color='black')
        ax.xaxis.set_major_locator(MultipleLocator(1))

        if settings.get('Rainbow'):
            colors = cm.rainbow(np.linspace(0, 1, segs))
            """------NORMAL-------"""
            for i in range(0, segs-1):
                ax.plot([i, i+2], [0, 0], color=colors[i])
            """------ACTUATED-------"""
            for i in range(0, segs-1):
                ax.plot(points[0, i:i+2], points[1, i:i+2], color=colors[i])
        else:
            """------NORMAL-------"""
            normal = np.zeros((16))
            ax.plot(normal, color='grey',
                    label="Initial pressure (P=P" + r'$_i$' + ")")
            """------ACTUATED-------"""
            ax.plot(points[0, :], points[1, :], color='red',
                    label="Final pressure (P=P" + r'$_f$' + ")")

        if any(value == True for value in parameters.get('Curve fitting').values()):
            m = points[0][-1]/(2*pi)
            if parameters.get('Curve fitting').get('Sin'):
                curve = np.sin(points[0]/m)
            elif parameters.get('Curve fitting').get('Cos'):
                curve = np.cos(points[0]/m)-1
            elif parameters.get('Curve fitting').get('Custom'):
                curve = []
                func = parameters.get('Curve fitting').get('Custom func')
                for ea in points[0]/m:
                    x = ea
                    curve.append(eval(func))

            ax.plot(points[0], curve, color='black', alpha=0.85, linestyle='--', label='Desired shape')
        
        ax.margins(0.5, 0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale(False)

        if settings.get('Plot boundaries'):
            to_tuple = [(x, y) for x, y in zip(limb.XY[0], limb.XY[1])]
            line_check = LineString(to_tuple)
            line_top = line_check.parallel_offset(1.9, side='left')
            line_bottom = line_check.parallel_offset(1.9, side='right')

            ax.plot(line_top.xy[0], line_top.xy[1])
            ax.plot(line_bottom.xy[0], line_bottom.xy[1])

        if settings.get('Overlay images'):
            def imshow_affine(ax, z, *args, **kwargs):
                im = ax.imshow(z, *args, **kwargs);
                _, x2, y1, _ = im.get_extent()
                im._image_skew_coordinate = (x2, y1)
                return im

            width = 2.57
            height = 3.9

            image_directory = os.path.dirname(
                os.path.realpath(__file__)) + '\\BOX.PNG'
            img = plt.imread(image_directory, format='png')

            cps = [[], []]
            for i in range(segs):
                cps[0].append((points[0][i] + points[0][i+1])/2)
                cps[1].append((points[1][i] + points[1][i+1])/2)
            cps = np.asarray(cps)

            for i in range(cps.shape[1]):
                img_show = imshow_affine(
                    ax,
                    img,
                    interpolation='none',
                    extent=[0, width, 0, height],
                )

                c_x, c_y = width/2, (1.75*cos(radians(11)))

                if limb.orient[i] == "TOP":
                    rot_angle = 180 + degrees(rots[i+1])
                elif limb.orient[i] == "BOTTOM":
                    rot_angle = degrees(rots[i+1])
                else:
                    rot_angle = degrees(rots[i+1])

                transform_data = (transforms.Affine2D()
                                  .rotate_deg_around(c_x, c_y, rot_angle)
                                  .translate((cps[0][i]-c_x), (cps[1][i]-c_y))
                                  + ax.transData)

                img_show.set_transform(transform_data)

    plt.tight_layout()
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    # fig.canvas.manager.window.raise_()


if __name__ == '__main__':
    """Genetic Algorithm for Soft Actuator 

    Parameters:
        Fitness Metric      --  Metric by which each individual in a population is evaluated (choose one by changing to True)
        Curve fitting       --  Curve to fit actuator shape to. Custom functions must use 'x' as variable. 
        Population size     --  Population size of the GA
        Stopping criteria   --  Determines when the GA stops (choose one by changing to True)
                                    Tolerance - Fitness metric difference between subsequent generations' best individual
                                    Maximum iterations - Run for X iterations
        Maximum iterations  --  If Stopping criteria = Maximum iterations; Set number of iterations here
        Tolerace            --  If Stopping criteria = Tolerance; Set tolerance here
        Patience            --  If Stopping criteria = Tolerance; Grace period (in iterations) before GA is terminated
        Number of segments  --  Number of segments in actuator ('None' or 'Integer'). If 'None' - becomes learnable parameter
        Selection           --  GA selection percentages or integers. NB: One method must be 'rest'
                                    Elite - can be int or float
                                    Random - can be int or float or range for scheduled decrease of randomness
                                        Scheduled Decrease format = (start percentage, end percentage, number of steps in schedule) i.e. (0.5, 0.05, 100)
                                    Mutation - can be int or float
                                    Crossover - can be int or float
        Top                 --  Number of top individuals to save
    """
    global parameters, settings

    def test(vec):
        plot_limb(vec)
        
    parameters = {
        'Fitness Metric': {
            'Maximise': True,
            'Metrics': {
                'X-coordinate': False,
                'Y-coordinate': True,
                'Distance from origin': False,
                'Curve': False,
            }
        },
        'Curve fitting': {
            'Sin': False,
            'Cos': False,
            'Custom': False,
            'Custom func': 'x**0.5', # x**2, x**0.5, 2**x
        },
        'Population size': 100,
        'Stopping criteria': {
            'Maximum iterations': True,
            'Tolerance': True,
        },
        'Maximum iterations': 500,
        'Tolerance': 1e-10,
        'Patience': 100, 
        'Number of segments': {
            'Type': 'Integer',
            'Number': 100,
        },
        'Selection': {
            'Elite': 1,
            'Random': (0.6, 0.05, 100),
            'Mutation': 0.25,
            'Crossover': 'rest',
        },
        'Top': 2,
        'Choices': ['BOTTOM', 'TOP'],
        'Segment width': 1,
    }
 
    settings = {
        'Multiprocessing': False,
        'Plot final': True,
        'Rainbow': False,
        'Overlay images': True,
        'Save data': False,
        'Plot boundaries': False,
    }

    # test(15*['TOP'])

    GA(parameters)
