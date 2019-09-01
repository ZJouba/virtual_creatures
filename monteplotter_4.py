import ast
import gc
import os
import pickle
import sys
import time
import tqdm
import tkinter as tk
from itertools import groupby
from math import cos, pi, radians, sin
from tkinter import filedialog
import multiprocessing
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import file_html
from bokeh.events import Tap
from bokeh.io import export_png, export_svgs, output_file, save, show
from bokeh.layouts import Spacer, column, row
from bokeh.models import ColumnDataSource, Label, Range1d, Select
from bokeh.models.glyphs import Patch
from bokeh.models.markers import Cross
from bokeh.models.widgets import Div, Paragraph, PreText
from bokeh.palettes import RdYlGn11 as palette
from bokeh.palettes import brewer
from bokeh.plotting import curdoc, figure, reset_output
from bokeh.resources import CDN
from bokeh.server.server import Server
from bokeh.transform import log_cmap
from descartes.patch import PolygonPatch
from nltk.util import ngrams
from shapely.geometry import LineString, MultiLineString
from shapely import ops
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate
from tornado.ioloop import IOLoop

from Tools.Gen_Tools import plot_creature

def preProcessing(allData):
    
    print('\n' + ('-' * 100))
    print('Preprocessing...')
    print('-' * 100 + '\n')

    if not isinstance(allData, pd.DataFrame):
        allData = pd.DataFrame(allData[1:], columns=allData[0])

    # allData.fillna(0, inplace=True)
    # allData.replace(np.inf, 0, inplace=True)

    # allData.drop_duplicates(subset=['L_string', 'Fitness'], inplace=True)
    # allData.reset_index(inplace=True)

    # if allData.shape[0] > 2000:
    #     allData = allData.sample(int(allData.shape[0]*0.5), weights='Fitness')

    try:
        allData['Angle'] = allData['Angle'].apply(lambda x: x*(180/pi))
    except:
        pass

    allData['% Overlap'] = allData.apply(
        lambda x: ((1 - x['Area']) / (x['Linestring'].length + 0.785)),
    axis=1)
    allData['Centroid_X'] = allData['Linestring'].apply(
        lambda x: x.centroid.x)
    allData['Centroid_Y'] = allData['Linestring'].apply(
        lambda x: x.centroid.y)
    allData['Compactness'] = allData['Bounds'].apply(
        lambda x: np.linalg.norm(x))
    allData['Length'] = allData['Linestring'].apply(
        lambda x: x.length)

    scaler = MinMaxScaler(feature_range=(0, 10))
    allData['S_Area'] = scaler.fit_transform(
        allData['Area'].values.reshape(-1, 1))

    sscaler = MinMaxScaler(feature_range=(0, 0.05))
    allData['S_Fitness'] = sscaler.fit_transform(
        allData['Fitness'].values.reshape(-1, 1))

    ngram_list = allData['L_string'].apply(lambda x: ((''.join(tup))
                                                      for i in range(2, 6) for tup in list(ngrams(list(x), i))))

    gram = []
    for ngram in ngram_list:
        tmp = [[*v] for _, v in groupby(sorted(ngram, key=len), key=len)]
        gram.append(tmp)

    allData['Rolling n-grams'] = gram

    gc.collect()

    for i in range(2, 6):
        allData['{}-gram'.format(i)] = allData['L_string'].apply(lambda x: [x[j:j+i]
                                                                            for j in range(0, len(x), i)])

    allData.to_csv("frameCSV.csv")

    return allData


def modify_doc(doc):
    """Add plots to the document

    Parameters
    ----------
    doc : bokeh.document.document.Document
        A Bokeh document to which plots can be added

    Returns
    ----------
        Document
    """

    plotData = allData.select_dtypes(include=np.number)

    scatter = ColumnDataSource(data=plotData)
    line = ColumnDataSource(data=dict(x=[], y=[]))
    r_1 = ColumnDataSource(data=dict(x=[], y=[]))
    r_2 = ColumnDataSource(data=dict(x=[], y=[]))
    polygon = ColumnDataSource(data=dict(x=[], y=[]))
    r_1_poly = ColumnDataSource(data=dict(x=[], y=[]))
    r_2_poly = ColumnDataSource(data=dict(x=[], y=[]))
    dist = ColumnDataSource(data=dict(x=[0], F=[0], P=[0], M=[0]))

    hist, edges = np.histogram(
        allData['S_Area'].values, bins='auto')
    dist_dict = ColumnDataSource(dict(
        hist=hist, edges_left=edges[:-1], edges_right=edges[1:]))

    palette.reverse()
    mapper = log_cmap(
        field_name='Area', palette=palette, low=0, high=500)

    tooltips1 = [
        ('index', '$index'),
        ('F', '@{PercentF}{0.0%}'),
        ('+', '@{Percent+}{0.0%}'),
        ('-', '@{Percent-}{0.0%}'),
        ('[', '@{Percent[}{0.0%}'),
        (']', '@{Percent]}{0.0%}'),
        ('X', '@{PercentX}{0.0%}'),
        ('_', '@{Percent_}{0.0%}'),
    ]
    tooltips2 = [
        ('index', '$index'),
        ('F', '@{MaxF}'),
        ('+', '@{Max+}'),
        ('-', '@{Max-}'),
    ]

    tips_angle = [
        ('Angle', '@Angle')
    ]

    """ Plots
    -----------------------------------------------------------------------------------------------------
    """
    scargs = {
        'size': 7,
        'source': scatter,
        'color': mapper,
        'alpha': 0.6,
        'nonselection_fill_color': mapper,
        'nonselection_fill_alpha': 0.1,
        'selection_fill_alpha': 1,
        'selection_fill_color': 'red',
    }

    plot_width = 450
    plot_height = 500
    fargs = {
        'plot_width': plot_width,
        'plot_height': plot_height,
        'tools': 'pan,wheel_zoom,box_zoom,reset,tap,save,box_select',
        'output_backend': 'webgl',
    }

    per_scatter = figure(**fargs, title="Fitness", tooltips=tooltips1)
    per_scatter.xaxis.axis_label = '% of character'
    per_scatter.yaxis.axis_label = 'Fitness'
    per_scatter.scatter('PercentF', 'Fitness', **scargs)

    seq_scatter = figure(**fargs, title="Area", tooltips=tooltips2)
    seq_scatter.xaxis.axis_label = 'Length of sequence'
    seq_scatter.yaxis.axis_label = 'Area'
    seq_scatter.scatter('MaxF', 'Area', **scargs)

    ang_scatter = figure(**fargs, title="Angle", tooltips=tips_angle)
    ang_scatter.scatter('Angle',  'Area', **scargs)
    ang_scatter.xaxis.axis_label = 'Angle (degrees)'
    ang_scatter.yaxis.axis_label = 'Area'

    rule_1_plot = figure(plot_width=300, plot_height=plot_height//2,
                         title="Rule 1", output_backend="webgl", match_aspect=True)
    rule_1_plot.patch(x='x', y='y', source=r_1_poly)
    rule_1_plot.line(x='x', y='y', line_color='red', source=r_1)

    rule_2_plot = figure(plot_width=300, plot_height=plot_height//2,
                         title="Rule 2", output_backend="webgl", match_aspect=True)
    rule_2_plot.patch(x='x', y='y', source=r_2_poly)
    rule_2_plot.line(x='x', y='y', line_color='red', source=r_2)

    char_F_dist = figure(plot_width=plot_width, plot_height=plot_height//2,
                         title="F character distribution", output_backend="webgl"
                         )
    char_F_dist.varea_stack('F', x='x', source=dist)
    char_F_dist.xaxis.axis_label = 'L_string length'

    char_M_dist = figure(plot_width=plot_width, plot_height=plot_height//2,
                         title="- character distribution", output_backend="webgl"
                         )
    char_M_dist.varea_stack('M', x='x', source=dist)
    char_M_dist.xaxis.axis_label = 'L_string length'

    char_P_dist = figure(plot_width=plot_width, plot_height=plot_height//2,
                         title="+ character distribution", output_backend="webgl"
                         )
    char_P_dist.varea_stack('P', x='x', source=dist)
    char_P_dist.xaxis.axis_label = 'L_string length'

    overlap_scatter = figure(**fargs, title="Overlap")
    overlap_scatter.xaxis.axis_label = 'Angle'
    overlap_scatter.yaxis.axis_label = '% Overlap'
    overlap_scatter.scatter('Angle', '% Overlap', **scargs)

    comp_scatter = figure(**fargs, title="Compactness")
    comp_scatter.xaxis.axis_label = 'Creature length'
    comp_scatter.yaxis.axis_label = 'Bounding box diagonal distance'
    comp_scatter.scatter('Length', 'Compactness', **scargs)

    centr_scatter = figure(
        **fargs, title="Centroid location", match_aspect=True)
    centr_scatter.xaxis.axis_label = 'X'
    centr_scatter.yaxis.axis_label = 'Y'
    centr_scatter.scatter('Centroid_X', 'Centroid_Y',
                          radius='S_Area', **scargs)

    comp_angle = figure(**fargs, title="Compactness")
    comp_angle.xaxis.axis_label = 'Angle'
    comp_angle.yaxis.axis_label = 'Bounding box diagonal distance'
    comp_angle.scatter('Angle', 'Compactness', **scargs)

    comp_char = figure(**fargs, title="Compactness")
    comp_char.xaxis.axis_label = '% of char'
    comp_char.yaxis.axis_label = 'Bounding box diagonal distance'
    # comp_char.scatter('Percent+', 'Compactness', **scargs)
    # comp_char.scatter('Percent-', 'Compactness', **scargs)

    comp_area = figure(**fargs, title="Compactness")
    comp_area.xaxis.axis_label = 'Area'
    comp_area.yaxis.axis_label = 'Bounding box diagonal distance'
    comp_area.scatter('Area', 'Compactness', **scargs)

    dist_plot = figure(plot_width=plot_width*3,
                       plot_height=plot_height,
                       title="Distributions",
                       output_backend="webgl",
                       x_axis_label='Select metric',
                       y_axis_label='Creatures'
                       )
    dist_select = Select(value=' ', title='Metric',
                         options=list(allData.select_dtypes(include=[np.number]).columns[1:].values))
    dist_plot.quad(top='hist', bottom=0, left='edges_left', right='edges_right',
                   fill_color="navy", line_color="white", alpha=0.5, source=dist_dict)

    branch_scatter = figure(
        **fargs, title="Branching chars")
    branch_scatter.xaxis.axis_label = '% of ['
    branch_scatter.yaxis.axis_label = '% of ]'
    branch_scatter.scatter('Percent[', 'Percent]',
                           radius='S_Fitness', **scargs)

    F_plot = figure(**fargs, title="Area")
    F_plot.xaxis.axis_label = 'No. of F'
    F_plot.yaxis.axis_label = 'Area'
    F_plot.scatter('CountF', 'Area', **scargs)

    scatter_select_x = Select(value='Area', title='Any plot x-value',
                         options=list(allData.select_dtypes(include=[np.number]).columns[1:].values))
    scatter_select_y = Select(value='Area', title='Any plot y-value',
                         options=list(allData.select_dtypes(include=[np.number]).columns[1:].values))
    scatter_select = figure(**fargs,
                            title="Any scatter",
                            x_axis_label='Select metric',
                            y_axis_label='Select metric'
                            )
    

    """ Text
    -----------------------------------------------------------------------------------------------------
    """
    L_string = Paragraph(text='Select creature', width=1200)
    characteristics = Div(text='Select creature', width=450)
    grams_static = PreText(text='Select creature', width=450)
    grams_rolling = PreText(text='Select creature', width=450)
    coordinates = PreText(text='Select creature', width=450)

    def clear():
        line.data = dict(x=[0, 0], y=[0, 0])
        polygon.data = dict(x=[0, 0], y=[0, 0])
        r_1.data = dict(x=[0, 0], y=[0, 0])
        r_2.data = dict(x=[0, 0], y=[0, 0])
        r_1_poly.data = dict(x=[0, 0], y=[0, 0])
        r_2_poly.data = dict(x=[0, 0], y=[0, 0])
        dist.data = dict(x=[0], F=[0], M=[0], P=[0])
        L_string.text = 'Select creature'
        characteristics.text = 'Select creature'
        grams_static.text = 'Select creature'
        grams_rolling.text = 'Select creature'
        coordinates.text ='Select creature'

    def to_coords(string, angle):
        """Converts rule to coordinates

        Parameters
        ----------
        string : String
            L_string of creature
        angle : Float
            Delta-angle (in radians) for creature

        Returns
        -------
        List
            Coordinate list representing L_string
        """

        num_chars = len(string)

        coords = np.zeros((num_chars + 1, 4), np.double)
        nodes = np.zeros((1, 4), np.double)

        rotVec = np.array((
            (cos(angle), -sin(angle), 0),
            (sin(angle), cos(angle), 0),
            (0, 0, 1)
        ))

        start_vec = np.array((0, 1, 0), np.float64)
        curr_vec = start_vec
        i = 1

        for c in string:
            """
            1: Node
            2: Branch
            3: Saved
            """
            if c == 'F':
                coords[i, :3] = (coords[i-1, :3] + (1 * curr_vec))
                i += 1

            if c == '-':
                curr_vec = np.dot(curr_vec, (-1*rotVec))

            if c == '+':
                curr_vec = np.dot(curr_vec, rotVec)

            if c == '[':
                nodes = np.vstack((nodes, coords[i-1]))
                coords[i-1, 3] = 3
                nodes[-1, 3] = 1

            if c == ']':
                if coords[i-1, 3] == 1:
                    # coords[i, 3] = 2
                    coords[i-1] = nodes[-1]
                    # i += 1
                else:
                    coords[i-1, 3] = 2
                    if len(nodes) == 1:
                        coords[i] = nodes[-1]
                    else:
                        value, nodes = nodes[-1], nodes[:-1]
                        coords[i] = value
                    i += 1

        coords = np.delete(coords, np.s_[i:], 0)
        return coords

    def plot_creature(event):
        """Plots creature and corresponding characteristics on datapoint select

        Parameters
        ----------
        event : bokeh.event
            On Tap event registered by Bokeh
        """

        def draw():
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # creature = ops.unary_union([
            #     ops.unary_union(creature_moves)] + [creature_linestring])

            creature_patch = PolygonPatch(creature['absorbA'], fc='BLACK', alpha=0.1)

            ax.add_patch(creature_patch)

            try:
                for line in creature_linestring:
                    x, y = line.xy
                    ax.plot(x, y, 'r-', zorder=1)
            except:
                x, y = creature_linestring.xy
                ax.plot(x, y, 'r-', zorder=1)

            for move in creature_moves:
                for line in move:
                    x, y = line.xy
                    ax.plot(x, y, 'g--', alpha=0.25)

            for p in env.patches:
                p = PolygonPatch(p)
                color = np.random.rand(3,)
                p.set_color(color)
                p.set_alpha(0.4)
                ax.add_patch(p)

            ax.autoscale(axis='y')
            ax.axis('equal')
            ax.plot(0, 0, 'xr')

            # plt.draw()
            # fig.canvas.draw_idle()
            # plt.pause(1)

        clear()

        if len(scatter.selected.indices) > 0:

            creature_index = scatter.selected.indices[0]
            creature = allData.iloc[creature_index, :]
            coords = creature['Coords']
            try:
                rules = creature['Rules']
                try:
                    rules = rules['X']
                    probas = rules['probabilities']
                    rules = rules['options']
                except:
                    probas = [0, 0]

                characteristics.text = 'Area:\t{:.2f}'.format(creature['Area']) + \
                    '</br>' + \
                    'Achievable creature area:\t{}'.format(creature['L_string'].count('F')+0.785) + \
                    '</br>' + \
                    'Overlap:\t{:.1%}'.format(1 - creature['Area'] / (creature['L_string'].count('F')+0.785)) + \
                    '</br>' + \
                    'Length of L_string:\t{}'.format(len(creature['L_string'])) + \
                    '</br>' + \
                    'Achievable maxmimum area:\t{}'.format(
                        len(creature['L_string']) + 0.785) + \
                    '</br>' + \
                    'Rule 1: <i><tab>{}</i>'.format(rules[0]) + \
                    '<tab><tab> Pr: <tab>{:.2%}'.format(probas[0]) + \
                    '</br>' + \
                    'Rule 2: <i><tab>{}</i>'.format(rules[1]) + \
                    '<tab><tab> Pr: <tab>{:.2%}'.format(probas[1])

            except:
                characteristics.text = 'Area:\t{:.2f}'.format(creature['Area']) + \
                    '</br>' + \
                    'Achievable creature area:\t{}'.format(creature['L_string'].count('F')+0.785) + \
                    '</br>' + \
                    'Overlap:\t{:.1%}'.format(1 - creature['Area'] / (creature['L_string'].count('F')+0.785)) + \
                    '</br>' + \
                    'Length of L_string:\t{}'.format(len(creature['L_string'])) + \
                    '</br>' + \
                    'Achievable maxmimum area:\t{}'.format(
                        len(creature['L_string']) + 0.785)

            L_string.text = '{}'.format(creature['L_string'])

            n_grams = {}
            for i in range(2, len(creature['Rolling n-grams'])):
                n_grams[str(i) + '-gram'] = creature[str(i) + '-gram']

            gram_frame_1 = pd.DataFrame.from_dict(n_grams, orient='index').T

            # gram_frame_1 = pd.DataFrame.from_dict(
            #     {'2-gram': creature['2-gram'],
            #      '3-gram': creature['3-gram'],
            #      '4-gram': creature['4-gram'],
            #      '5-gram': creature['5-gram'],
            #      },
            #     orient='index').T

            counts = [pd.value_counts(gram_frame_1[i]).reset_index().astype(
                str).apply(' '.join, 1) for i in gram_frame_1]
            out = pd.concat(counts, 1).fillna('')
            out.columns = gram_frame_1.columns
            grams_static.text = ('-' * 14) + ' Static n-grams ' + ('-' * 14) + '\n' + str(
                tabulate(out, headers='keys'))

            n_grams = {}
            for i in range(len(creature['Rolling n-grams'])):
                n_grams[str(i) + '-gram'] = creature['Rolling n-grams'][i]

            gram_frame_2 = pd.DataFrame.from_dict(n_grams, orient='index').T

            counts = [pd.value_counts(gram_frame_2[i]).reset_index().astype(
                str).apply(' '.join, 1) for i in gram_frame_1]
            out = pd.concat(counts, 1).fillna('')
            out.columns = gram_frame_1.columns
            grams_rolling.text = ('-' * 14) + ' Rolling n-grams ' + ('-' * 14) + '\n' + str(
                tabulate(out, headers='keys'))

            coordinates.text = str(
                tabulate(creature['Coords'], headers='keys'))

            creature_linestring = creature['Linestring']
            creature_moves = creature['moves']

            if 'F' in rules[0]:
                r_1_coords = to_coords(rules[0], creature['Angle'])

                r_1_morphology = LineString(r_1_coords[:, 0:2])

                r_1_poly.data = dict(
                    x=r_1_morphology.buffer(0.5).exterior.coords.xy[0],
                    y=r_1_morphology.buffer(0.5).exterior.coords.xy[1],
                )
                r_1.data = dict(x=r_1_coords[:, 0], y=r_1_coords[:, 1])

            if 'F' in rules[1]:
                r_2_coords = to_coords(rules[1], creature['Angle'])

                r_2_morphology = LineString(r_2_coords[:, 0:2])

                r_2_poly.data = dict(
                    x=r_2_morphology.buffer(0.5).exterior.coords.xy[0],
                    y=r_2_morphology.buffer(0.5).exterior.coords.xy[1],
                )
                r_2.data = dict(x=r_2_coords[:, 0], y=r_2_coords[:, 1])

            c_string = creature['L_string']
            bins_width = 10
            bins = int(len(c_string)//bins_width)
            dists = {}
            dists['F'] = []
            dists['P'] = []
            dists['M'] = []
            dists['x'] = []

            for i in range(bins):
                start = ((i-1)*bins_width)
                end = (i*bins_width)-1
                dists['F'].append(c_string.count('F', start, end))
                dists['P'].append(c_string.count('+', start, end))
                dists['M'].append(c_string.count('-', start, end))
            [dists['x'].append(i) for i in range(1, bins+1)]
            dist.data = dists

            draw()
            plt.pause(10)
            # plt.show()

        else:
            clear()
            plt.close('all')

    def update_dist(attrname, old, new):
        scaler = StandardScaler()
        name = dist_select.value
        hist, edges = np.histogram(
            scaler.fit_transform(
                allData[name].values.reshape(-1, 1)), bins=int(allData.shape[0]/10))
        dist_dict.data = dict(
            hist=hist, edges_left=edges[:-1], edges_right=edges[1:])
        dist_plot.xaxis.axis_label = name
    
    def update_scatter(attrname, old, new):
        scatter_select.renderers = []
        scatter_select.scatter(
            x=scatter_select_x.value, y=scatter_select_y.value, **scargs)
        scatter_select.xaxis.axis_label = scatter_select_x.value
        scatter_select.yaxis.axis_label = scatter_select_y.value


    per_scatter.on_event(Tap, plot_creature)
    seq_scatter.on_event(Tap, plot_creature)
    ang_scatter.on_event(Tap, plot_creature)
    overlap_scatter.on_event(Tap, plot_creature)
    comp_scatter.on_event(Tap, plot_creature)
    centr_scatter.on_event(Tap, plot_creature)
    comp_angle.on_event(Tap, plot_creature)
    comp_char.on_event(Tap, plot_creature)
    comp_area.on_event(Tap, plot_creature)
    F_plot.on_event(Tap, plot_creature)
    dist_select.on_change('value', update_dist)
    scatter_select_x.on_change('value', update_scatter)
    scatter_select_y.on_change('value', update_scatter)
    branch_scatter.on_event(Tap, plot_creature)
    scatter_select.on_event(Tap, plot_creature)

    row_A = row(L_string)
    row_B = row(per_scatter, seq_scatter, ang_scatter, overlap_scatter)
    row_C_right = column(rule_1_plot, rule_2_plot)
    row_C_middle = row(grams_rolling, grams_static)
    row_C = row(
        characteristics,
        Spacer(width=50),
        row_C_middle,
        Spacer(width=50),
        row_C_right)
    row_D = row(char_F_dist, char_M_dist, char_P_dist)
    row_E = row(comp_scatter, comp_angle, comp_char, centr_scatter)
    row_F = row(comp_area, F_plot, branch_scatter, coordinates)
    row_G = column(dist_select, dist_plot)
    row_H = column(scatter_select_x, scatter_select_y, scatter_select)

    layout = column(
        row_A,
        row_B,
        row_C,
        row_D,
        row_E,
        row_F,
        row_G,
        row_H,
    )

    clear()
    doc.add_root(layout)


def main():
    """Launch bokeh server and connect to it
    """
    print('\n' + ('-' * 100))
    print('Select file...')
    print('-' * 100 + '\n')

    global allData, env

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filepath = filedialog.askopenfilename()

    if filepath:
        allData = pickle.load(open(filepath, 'rb'))

        env_path = os.path.dirname(filepath) + '/Environment ' + filepath[-22:]
        env = pickle.load(open(env_path, 'rb'))
        
        # allData = tqdm.tqdm(preProcessing(allData), total=allData.shape[0], file=sys.stdout)
        allData = preProcessing(allData)

        print('\n' + ('-' * 100))
        print('ALL DONE!')
        print('-' * 100 + '\n')
        print("Preparing a bokeh application.")
        io_loop = IOLoop.current()
        bokeh_app = Application(FunctionHandler(modify_doc))

        server = Server({'/app': bokeh_app}, io_loop=io_loop, port=5001)
        server.start()
        print("Opening Bokeh application on http://localhost:5006/")
        server.show('/app')
        io_loop.start()


main()
