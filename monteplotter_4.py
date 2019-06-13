from bokeh.io import output_file, show, export_png, export_svgs, save
from bokeh.plotting import figure, curdoc, reset_output
from bokeh.models import ColumnDataSource, Label, Range1d
from bokeh.layouts import row, column, Spacer
from bokeh.models.glyphs import Patch
from bokeh.models.widgets import PreText, Paragraph, Div
from bokeh.models.markers import Cross
from bokeh.events import Tap
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
from bokeh.transform import log_cmap
from bokeh.palettes import brewer
from bokeh.palettes import RdYlGn11 as palette
from bokeh.embed import file_html
from bokeh.resources import CDN
from tornado.ioloop import IOLoop
import pandas as pd
from shapely.geometry import LineString
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from tkinter import filedialog
import tkinter as tk
import os
import gc
import ast
import sys
from math import radians, cos, sin, pi
from tabulate import tabulate
import time
import pickle


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
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()

    allData = pickle.load(open(filepath, 'rb'))

    allData.fillna(0, inplace=True)
    allData.replace(np.inf, 0, inplace=True)

    allData.drop_duplicates(subset='L-string', inplace=True)
    allData['Angle'] = allData['Angle'].apply(lambda x: x*(180/pi))
    overlap = []
    linestrings = []
    for _, creature in allData.iterrows():
        coords = creature['Coordinates']
        linestrings.append(LineString(coords[:, 0:2]))
        overlap.append(1 - creature['Area'] /
                       (linestrings[-1].length+0.785))

    allData['Line Strings'] = linestrings
    allData['% Overlap'] = overlap
    cr = allData.iloc[1, :]
    a = cr['Line Strings'].centroid.xy
    print(a[0])
    allData['Centroid_X'] = allData['Line Strings'].apply(
        lambda x: x.centroid.coords.xy[0].value)
    allData['Centroid_Y'] = allData['Line Strings'].apply(
        lambda x: x.centroid.coords.xy[1])
    allData['Compactness'] = allData['Bounding Coordinates'].apply(
        lambda x: np.linalg.norm(x))

    for i in range(2, 6):
        allData['{}-gram'.format(i)] = allData['L-string'].apply(lambda x: [x[j:j+i]
                                                                            for j in range(0, len(x), i)])

    gc.collect()

    plotData = allData.iloc[:, 1:]
    # plotData = allData.select_dtypes(include=['float64', 'int64'])

    scatter = ColumnDataSource(data=plotData)
    line = ColumnDataSource(data=dict(x=[], y=[]))
    r_1 = ColumnDataSource(data=dict(x=[], y=[]))
    r_2 = ColumnDataSource(data=dict(x=[], y=[]))
    polygon = ColumnDataSource(data=dict(x=[], y=[]))
    r_1_poly = ColumnDataSource(data=dict(x=[], y=[]))
    r_2_poly = ColumnDataSource(data=dict(x=[], y=[]))
    dist = ColumnDataSource(data=dict(x=[], y=[]))

    palette.reverse()
    mapper = log_cmap(
        field_name='Area', palette=palette, low=0, high=500)

    tooltips1 = [
        ('index', '$index'),
        ('F', '@{% of F}{0.0%}'),
        ('+', '@{% of +}{0.0%}'),
        ('-', '@{% of -}{0.0%}'),
    ]
    tooltips2 = [
        ('index', '$index'),
        ('F', '@{Longest F sequence}'),
        ('+', '@{Longest + sequence}'),
        ('-', '@{Longest - sequence}'),
    ]

    tips_angle = [
        ('Angle', '@Angle')
    ]

    """ Plots
    -----------------------------------------------------------------------------------------------------
    """
    plots_width = 500
    plots_height = 500
    per_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                         title="Area", output_backend="webgl", tooltips=tooltips1)
    per_scatter.xaxis.axis_label = '% of character'
    per_scatter.yaxis.axis_label = 'Area'
    per_scatter.scatter('% of F', 'Area', size=7,
                        source=scatter, color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    seq_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                         title="Area", output_backend="webgl", tooltips=tooltips2)
    seq_scatter.xaxis.axis_label = 'Length of sequence'
    seq_scatter.yaxis.axis_label = 'Area'
    seq_scatter.scatter('Longest F sequence', 'Area', size=7, source=scatter,
                        fill_color='red', color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    creature_plot = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                           title="Selected Creature", output_backend="webgl")
    creature_plot.axis.visible = False
    creature_plot.grid.visible = False
    creature_plot.multi_polygons(xs='x', ys='y', source=polygon)
    creature_plot.line(x='x', y='y', line_color='red', source=line)
    start_point = Cross(x=0, y=0, size=10, line_color='red', line_width=5)
    creature_plot.add_glyph(start_point)

    ang_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                         title="Angle", output_backend="webgl", tooltips=tips_angle)
    ang_scatter.scatter('Angle',  'Area',  size=7,
                        source=scatter, color=mapper, alpha=0.6, nonselection_fill_color=mapper)
    ang_scatter.xaxis.axis_label = 'Angle (degrees)'
    ang_scatter.yaxis.axis_label = 'Area'

    rule_1_plot = figure(plot_width=300, plot_height=plots_height//2,
                         title="Rule 1", output_backend="webgl")
    rule_1_plot.patch(x='x', y='y', source=r_1_poly)
    rule_1_plot.line(x='x', y='y', line_color='red', source=r_1)

    rule_2_plot = figure(plot_width=300, plot_height=plots_height//2,
                         title="Rule 2", output_backend="webgl")
    rule_2_plot.patch(x='x', y='y', source=r_2_poly)
    rule_2_plot.line(x='x', y='y', line_color='red', source=r_2)

    char_dist = figure(plot_width=plots_width, plot_height=plots_height//2,
                       title="Character distribution", output_backend="webgl"
                       )
    char_dist.varea_stack(['F', 'P', 'M'], x='x',
                          color=brewer['Spectral'][3], legend=['F char', '+ char', '- char'], source=dist)
    char_dist.xaxis.axis_label = 'L-string length'

    overlap_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                             title="Overlap", output_backend="webgl")
    overlap_scatter.xaxis.axis_label = 'Angle'
    overlap_scatter.yaxis.axis_label = '% Overlap'
    overlap_scatter.scatter('Angle', '% Overlap', size=7, source=scatter,
                            fill_color='red', color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    comp_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                          title="Compactness", output_backend="webgl")
    comp_scatter.xaxis.axis_label = 'Index'
    comp_scatter.yaxis.axis_label = 'Compactness'
    comp_scatter.scatter('index', 'Compactness', size=7, source=scatter,
                         fill_color='red', color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    centr_scatter = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                           title="Centroid location", output_backend="webgl")
    centr_scatter.xaxis.axis_label = 'X'
    centr_scatter.yaxis.axis_label = 'Y'
    centr_scatter.scatter('Centroid'[0], 'Centroid'[1], size=7, source=scatter,
                          fill_color='red', color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    """ Text
    -----------------------------------------------------------------------------------------------------
    """
    L_string = Paragraph(text='Select creature', width=1200)
    characteristics = Div(text='Select creature', width=200)
    grams = PreText(text='Select creature', width=400)
    rule_text = PreText(text='Select creature', width=400)

    def to_coords(string, angle):
        """Converts L-string to coordinates

        Parameters
        ----------
        string : String
            L-string of creature
        angle : Float
            Delta-angle (in radians) for creature

        Returns
        -------
        List
            Coordinate list representing L-string
        """

        num_chars = len(string)

        coords = np.zeros((num_chars + 1, 3), np.double)

        rotVec = np.array((
            (cos(angle), -sin(angle), 0),
            (sin(angle), cos(angle), 0),
            (0, 0, 1)
        ))

        start_vec = np.array((0, 1, 0), np.float64)
        curr_vec = start_vec
        i = 1

        for c in string:
            if c == 'F':
                coords[i] = coords[i-1] + curr_vec
                i += 1

            if c == '-':
                curr_vec = np.dot(curr_vec, (-1*rotVec))

            if c == '+':
                curr_vec = np.dot(curr_vec, rotVec)

        coords = np.delete(coords, np.s_[i:], 0)
        return coords

    def plot_creature(event):
        """Plots creature and corresponding characteristics on datapoint select

        Parameters
        ----------
        event : bokeh.event
            On Tap event registered by Bokeh
        """

        def clear():
            line.data = dict(x=[0, 0], y=[0, 0])
            polygon.data = dict(x=[0, 0], y=[0, 0])
            r_1.data = dict(x=[0, 0], y=[0, 0])
            r_2.data = dict(x=[0, 0], y=[0, 0])
            r_1_poly.data = dict(x=[0, 0], y=[0, 0])
            r_2_poly.data = dict(x=[0, 0], y=[0, 0])
            dist.data = dict(x=[0], F=[0], M=[0], P=[0])
            L_string.text = 'Select creature'
            rule_text.text = 'Select creature'
            characteristics.text = 'Select creature'

        clear()

        if len(scatter.selected.indices) > 0:

            creature_index = scatter.selected.indices[0]
            creature = allData.iloc[creature_index, :]
            coords = creature['Coordinates']
            # coords = np.array(ast.literal_eval(creature['Coordinates']))

            L_string.text = '{}'.format(creature['L-string'])

            characteristics.text = 'Area:\t{:.2f}'.format(creature['Area']) + \
                '</br>' + \
                'Length of L-string:\t{}'.format(len(creature['L-string'])) + \
                '</br>' + \
                'Maximum string area:\t{}'.format(len(creature['L-string']) + 0.785) + \
                '</br>' + \
                'Maximum creature area:\t{}'.format(creature['L-string'].count('F')+0.785) + \
                '</br>' + \
                'Overlap:\t{:.1%}'.format(1 -
                                          creature['Area'] / (creature['L-string'].count('F')+0.785))

            gram_frame_1 = pd.DataFrame.from_dict(
                {'2-gram': creature['2-gram'],
                 '3-gram': creature['3-gram'],
                 '4-gram': creature['4-gram'],
                 '5-gram': creature['5-gram'],
                 },
                orient='index').T

            counts = [pd.value_counts(gram_frame_1[i]).reset_index().astype(
                str).apply(' '.join, 1) for i in gram_frame_1]
            out = pd.concat(counts, 1).fillna('')
            out.columns = gram_frame_1.columns
            grams.text = str(
                tabulate(out, headers='keys'))

            creature_linestring = LineString(coords[:, 0:2])
            creature_patch = creature_linestring.buffer(0.5)
            patch_x, patch_y = creature_patch.exterior.coords.xy

            x_points = [list(patch_x)]
            y_points = [list(patch_y)]

            for i, _ in enumerate(creature_patch.interiors):
                x_in, y_in = creature_patch.interiors[i].coords.xy
                x_points.append(list(x_in))
                y_points.append(list(y_in))

            x_points = [[x_points]]
            y_points = [[y_points]]

            line.data = dict(x=coords[:, 0], y=coords[:, 1])
            polygon.data = dict(x=x_points, y=y_points)

            # rules = ast.literal_eval(creature['Rules'])
            rules = creature['Rules']
            rules = rules['X']
            probas = rules['probabilities']
            rules = rules['options']

            rule_text.text = 'Rule 1: \t{}'.format(rules[0]) + \
                '\t Pr: \t{:.2%}'.format(probas[0]) + \
                '\n' + \
                'Rule 2: \t{}'.format(rules[1]) + \
                '\t Pr: \t{:.2%}'.format(probas[1])

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

            c_string = creature['L-string']
            bins_width = 25
            bins = len(c_string)//bins_width
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
            char_dist.x_range = Range1d(-1, (bins+15))

        else:
            clear()

    per_scatter.on_event(Tap, plot_creature)
    seq_scatter.on_event(Tap, plot_creature)
    ang_scatter.on_event(Tap, plot_creature)
    overlap_scatter.on_event(Tap, plot_creature)
    comp_scatter.on_event(Tap, plot_creature)
    centr_scatter.on_event(Tap, plot_creature)

    row_A = row(L_string)
    row_B = row(per_scatter, seq_scatter, ang_scatter, overlap_scatter)
    row_C_right = column(rule_1_plot, rule_2_plot)
    row_C_middle = column(grams, rule_text)
    row_C = row(
        characteristics,
        creature_plot,
        Spacer(width=50),
        row_C_middle,
        Spacer(width=50),
        row_C_right)
    row_D = row(char_dist)
    row_E = row(comp_scatter, centr_scatter)

    layout = column(
        row_A,
        row_B,
        row_C,
        # row_D,
        row_E,
    )

    doc.add_root(layout)


def main():
    """Launch bokeh server and connect to it
    """

    print("Preparing a bokeh application.")
    io_loop = IOLoop.current()
    bokeh_app = Application(FunctionHandler(modify_doc))

    server = Server({'/app': bokeh_app}, io_loop=io_loop, port=5001)
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")
    server.show('/app')
    io_loop.start()


main()
