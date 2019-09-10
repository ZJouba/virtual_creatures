import itertools
import multiprocessing as mp
import os
import sys
from datetime import datetime
from decimal import Decimal
from math import atan2, cos, degrees, radians

import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from tabulate import tabulate

from grid_strategy import strategies
from Tools.Classes import Limb

from shapely.geometry import LineString


def evaluate(orient_vector):
    invalid = False

    l = Limb(len(orient_vector))

    l.build(orient_vector)

    origin = np.array((0, 0))
    dists = [np.linalg.norm(b-origin) for b in l.XY.T]
    D = max(dists)
    X = max(abs(l.XY[0]))
    Y = max(abs(l.XY[1]))

    Curvature = degrees(l.curvature[-1])

    limb_res = [
        round(X, 2),
        round(Y, 2),
        round(D, 2),
        round(Curvature, 2),
        l,
        orient_vector
    ]

    global top_X
    if min(top_X[:, sort_by]) < limb_res[sort_by] < max(top_X[:, sort_by]):
        to_tuple = [(x, y) for x, y in zip(l.XY[0], l.XY[1])]
        line_check = LineString(to_tuple)
        line_top = line_check.parallel_offset(1.9, side='left')
        line_bottom = line_check.parallel_offset(1.9, side='left')

        if not line_check.is_simple or not line_top.is_simple or not line_bottom.is_simple:
            invalid = True
        if line_check.is_closed or line_top.is_closed or line_bottom.is_closed:
            invalid = True

    if invalid:
        D = 0
        X = 0
        Y = 0

        Curvature = 0

        limb_res = [
            round(X, 2),
            round(Y, 2),
            round(D, 2),
            round(Curvature, 2),
            l,
            orient_vector
        ]

    return limb_res


def selection(generation_data, num_segments, elite, randomised, mutation, crossover):
    global random_array

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
        new_generation_data.append(generation_data[i][5])

    if isinstance(num_segments, int):
        for i in range(randomised):
            orient = [np.random.choice(["TOP", "BOTTOM", "EMPTY"])
                      for _ in range(num_segments)]
            new_generation_data.append(orient)
    else:
        for i in range(randomised):
            top_margin = len(elite[0]) + 5
            bottom_margin = len(elite[0]) - 5
            orient = [np.random.choice(
                ["TOP", "BOTTOM", "EMPTY"]) for _ in range(np.random.randint(bottom_margin, top_margin))]
            new_generation_data.append(orient)

    for i in range(mutation):
        limb = generation_data[np.random.randint(len(generation_data))][5]
        index = np.random.randint(
            low=0, high=len(limb), size=int(len(limb)*0.2))
        for place in index:
            limb[place] = np.random.choice(["TOP", "BOTTOM", "EMPTY"])
        new_generation_data.append(limb)

    if crossover == 'rest':
        while len(new_generation_data) < len(generation_data):
            parentA = generation_data[np.random.randint(
                len(generation_data))][5]
            parentB = generation_data[np.random.randint(
                len(generation_data))][5]
            mid = int(len(parentA) * 0.5)
            child = parentA[:mid] + parentB[mid:]
            new_generation_data.append(child)
    else:
        for i in range(crossover):
            parentA = generation_data[np.random.randint(
                len(generation_data))][5]
            parentB = generation_data[np.random.randint(
                len(generation_data))][5]
            mid = int(len(parentA) * 0.5)
            child = parentA[:mid] + parentB[mid:]
            new_generation_data.append(child)

    return new_generation_data


def GA(parameters):
    def delete_lines(n=1):
        for _ in range(n):
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')

    i = 0

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
        "LIMB OBJECT",
        "ORIENTATION VECTOR",
    ]
    results = [[
        columns[0],
        columns[1],
        columns[2],
        columns[3],
        columns[4],
        columns[5],
    ]]

    if settings.get('Save data'):
        generation_save_directory.write(("{}\n".format(results[0])))

    pop_size = parameters.get('Population size')
    fitness = parameters.get('Fitness Metric')
    criteria = parameters.get('Stopping criteria')
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

    metric = [i for i, x in fitness.items() if x]
    if len(metric) > 1:
        exception_string = ('Only one fitness metric allowed')
        raise Exception(exception_string)

    global sort_by
    sort_by = list(fitness.keys()).index(metric[0])

    metric = [i for i, x in criteria.items() if x]
    if len(metric) > 1:
        exception_string = ('Only one stopping criteria allowed')
        raise Exception(exception_string)

    check_set = set(str(value).lower() for value in parameters.get(
        'Selection').values())
    if not 'rest' in check_set:
        exception_string = ('One selection method must be set to \'rest\'')
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
        exception_string = ('Cummulative selection values cannot exceed 100%')
        raise Exception(exception_string)

    stop_crit = metric[0]

    global top_X
    top_X = np.zeros((5, 5))

    while len(results) <= pop_size:
        if num_segments == 0:
            orientations = [np.random.choice(
                ["TOP", "BOTTOM", "EMPTY"]) for _ in range(np.random.randint(1, 50))]
        else:
            orientations = [np.random.choice(
                ["TOP", "BOTTOM", "EMPTY"]) for _ in range(num_segments)]

        if not orientations[0] == 'EMPTY':
            results.append(evaluate(orientations))

    results[1:] = sorted(results[1:], key=lambda x: x[sort_by], reverse=True)

    best = 0
    top = parameters.get('Top')
    # global top_X
    top_X = np.array(results[1:top+1])
    prev_gen_top = results[1][sort_by]
    stop_criteria_counter = 0
    stop = False
    iteration = 0

    while not stop:
        new_generation = selection(
            results[1:], num_segments, elite, rand, mutation, crossover)
        results = [["MAX X", "MAX Y", "MAX DISTANCE",
                    "TOTAL CURVATURE", "LIMB OBJECT", "ORIENTATION VECTOR"]]

        if settings.get('Multiprocessing'):
            with mp.Pool(mp.cpu_count()) as pool:
                mp_get = list(pool.imap_unordered(evaluate, new_generation))
            results += mp_get
        else:
            for ea in new_generation:
                results.append(evaluate(ea))

        results[1:] = sorted(
            results[1:], key=lambda x: x[sort_by], reverse=True)

        this_gen_top = results[1][sort_by]

        new_place = top-1
        if not this_gen_top in top_X[:, sort_by]:
            for place in range(top-1, 0, -1):
                if this_gen_top > top_X[place, sort_by]:
                    new_place -= 1
                    if new_place == 0:
                        break

            for i in range(top-1, new_place, -1):
                top_X[i] = top_X[i-1]

            top_X[new_place] = results[1]

        if this_gen_top > best:
            best_limb = results[1]
            best = results[1][sort_by]

        if stop_crit == 'Maximum iterations':
            stop_criteria_counter += 1
            if stop_criteria_counter > maximum:
                stop = True
        elif stop_crit == 'Tolerance':
            tol = abs(prev_gen_top - this_gen_top)
            if tol < tolerance:
                stop_criteria_counter += 1
            else:
                stop_criteria_counter = 0
            if stop_criteria_counter > patience:
                stop = True

        prev_gen_top = results[1][sort_by]
        iteration += 1

        print('\nIteration:\t{}'.format(iteration))
        print('\nTolerance:\t{:.10E}'.format(Decimal(tol)))

        delete_lines(n=4)

        if settings.get('Save data'):
            for line in results[1:]:
                generation_save_directory.write(("{}\n".format(line)))

    print("\n" + 200*"-")
    print("\nSTOPPING CRITERIA REACHED")
    print("\n" + 200*"-")

    if settings.get('Save data'):
        generation_save_directory.close()

    if settings.get('Plot final'):
        plot_limb(top_X[:, 4])


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

    num_plots = len(limbs)

    specs = strategies.SquareStrategy('center').get_grid(num_plots)

    fig = plt.figure(1, constrained_layout=False)
    fig.canvas.set_window_title('Top ' + str(num_plots))

    for limb, sub in zip(limbs, specs):
        ax = fig.add_subplot(sub)

        segs = limb.XY.shape[1]

        points = limb.XY
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
                ax.plot([i, i+1], [0, 0], color=colors[i])
            """------ACTUATED-------"""
            for i in range(0, segs-1):
                ax.plot(points[0, i:i+2], points[1, i:i+2], color=colors[i])
        else:
            """------NORMAL-------"""
            normal = np.zeros((15))
            ax.plot(normal, color='grey',
                    label="Initial pressure (P=P" + r'$_i$' + ")")
            """------ACTUATED-------"""
            ax.plot(points[0, :], points[1, :], color='red',
                    label="Final pressure (P=P" + r'$_f$' + ")")

        ax.margins(0.5, 0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale(False)

        if settings.get('Overlay images'):
            def imshow_affine(ax, z, *args, **kwargs):
                im = ax.imshow(z, *args, **kwargs)
                x1, x2, y1, y2 = im.get_extent()
                im._image_skew_coordinate = (x2, y1)
                return im

            width = 2.57
            height = 3.9

            image_directory = os.path.dirname(
                os.path.realpath(__file__)) + '\\box1.PNG'
            img = plt.imread(image_directory, format='png')

            cps = [[], []]
            for i in range(points.shape[1]-1):
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

                c_x, c_y = width/2, 1.7

                if limb.orient[i] == "TOP":
                    angle = 180 + degrees(rots[i+1])
                elif limb.orient[i] == "BOTTOM":
                    angle = degrees(rots[i+1])
                else:
                    angle = degrees(rots[i+1])

                transform_data = (transforms.Affine2D()
                                  .rotate_deg_around(c_x, c_y, angle)
                                  .translate((cps[0][i]-c_x), (cps[1][i]-c_y))
                                  + ax.transData)

                img_show.set_transform(transform_data)

    for m in limbs:
        print(tabulate(m.XY))
        print(m.orient)

    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    fig.canvas.manager.window.raise_()


if __name__ == '__main__':
    """Genetic Algorithm for Soft Actuator 

    Parameters:
        Fitness Metric      --  Metric by which each individual in a population is evaluated (choose one by changing to True)
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

    parameters = {
        'Fitness Metric': {
            'Maximum X-coordinate': False,
            'Maximum Y-coordinate': False,
            'Maximum distance from origin': False,
            'Maximum curve': True,
        },
        'Population size': 500,
        'Stopping criteria': {
            'Maximum iterations': False,
            'Tolerance': True,
        },
        'Maximum iterations': 100,
        'Tolerance': 1e-5,
        'Patience': 500,
        'Number of segments': {
            'Type': 'Integer',
            'Number': 15,
        },
        'Selection': {
            'Elite': 1,
            'Random': (0.6, 0.05, 200),
            'Mutation': 0.05,
            'Crossover': 'rest',
        },
        'Top': 6,
    }

    global settings
    settings = {
        'Multiprocessing': False,
        'Plot final': True,
        'Rainbow': False,
        'Overlay images': True,
        'Save data': False,
    }

    GA(parameters)
