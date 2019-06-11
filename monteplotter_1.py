from bokeh.io import output_file, show, export_png, export_svgs, save
from bokeh.plotting import figure, curdoc, reset_output
from bokeh.models import ColumnDataSource, HoverTool, Label, BoxZoomTool, ResetTool
from bokeh.layouts import row, column, Spacer
from bokeh.models.glyphs import Patch
from bokeh.models.widgets import PreText, Paragraph
from bokeh.events import Tap
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
import pandas as pd
from shapely.geometry import LineString
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
from tkinter import filedialog
import tkinter as tk
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from grid_strategy import strategies
import gc
import ast
from descartes.patch import PolygonPatch
import sys
from tabulate import tabulate
from math import radians, cos, sin


def modify_doc(doc):
    curDir = os.path.dirname(__file__)
    root = tk.Tk()
    root.withdraw()

    ProgressBar().register()

    filepath = filedialog.askopenfilename()
    filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]

    monteData = dd.read_csv(filepath)

    monteData.fillna(0)

    # plotData = monteData.sample(frac=0.01)
    plotData = monteData.compute()
    # plotData = plotData.compute()

    gc.collect()

    # plotData = pd.read_csv(os.path.join(curDir, 'test.csv'))

    plotData.drop_duplicates(subset='L-string', inplace=True)
    plotData.reset_index()
    for i in range(2, 6):
        plotData['{}-gram'.format(i)] = plotData['L-string'].apply(lambda x: [x[j:j+i]
                                                                              for j in range(0, len(x), i)])

    gc.collect()

    source = ColumnDataSource(data=plotData)
    source1 = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    source2 = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    source3 = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    rule1 = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    rule2 = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    # x_points = [0]
    # y_points = [0]

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

    plots_width = 500
    plots_height = 500
    p1 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl", tooltips=tooltips1)
    p1.xaxis.axis_label = 'Area'
    p1.yaxis.axis_label = '% of character'
    p1.scatter('Area', '% of F', size=7, source=source, alpha=0.6)

    p2 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl", tooltips=tooltips2)
    p2.xaxis.axis_label = 'Area'
    p2.yaxis.axis_label = 'Length of sequence'
    p2.scatter('Area', 'Longest F sequence', size=7, source=source,
               fill_color='red', legend='F', alpha=0.6)
    # p2.scatter('Area', 'Longest + sequence', size=7, source=source,
    #            fill_color='blue', legend='+', alpha=0.6)
    # p2.scatter('Area', 'Longest - sequence', size=7, source=source,
    #            fill_color='green', legend='-', alpha=0.6)

    p3 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Selected Creature", output_backend="webgl")
    p3.axis.visible = False
    p3.grid.visible = False
    p3.line(x='x', y='y', line_color='red', source=source1)
    # p3.multi_polygons(xs='x', ys='y', source=source2)
    # p3.multi_polygons(xs=x_points, ys=y_points)
    p3.patch(x='x', y='y', source=source2)
    p3.patch(x='x', y='y', source=source3, fill_color='white')

    p4 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl")
    p4.scatter('Area', 'Angle', size=7, source=source, alpha=0.6)
    p4.xaxis.axis_label = 'Area'
    p4.yaxis.axis_label = 'Angle (degrees)'

    p5 = figure(plot_width=plots_width, plot_height=plots_height//2,
                title="Rule 1", output_backend="webgl")
    p5.line(x='x', y='y', line_color='red', source=rule1)
    p6 = figure(plot_width=plots_width, plot_height=plots_height//2,
                title="Rule 2", output_backend="webgl")
    p6.line(x='x', y='y', line_color='red', source=rule2)

    L_string = Paragraph(text='Select creature', width=1500)

    grams = PreText(text='Select creature', width=400)
    rule_text = PreText(text='Select creature', width=400)

    area_label = Label(
        x=0,
        y=405,
        x_units='screen',
        y_units='screen',
        text='Select creature',
        render_mode='css',
        border_line_color='black',
        border_line_alpha=1.0,
        background_fill_color='white',
        background_fill_alpha=1.0,
    )

    length_label = Label(
        x=0,
        y=385,
        x_units='screen',
        y_units='screen',
        text='Select creature',
        render_mode='css',
        border_line_color='black',
        border_line_alpha=1.0,
        background_fill_color='white',
        background_fill_alpha=1.0,
    )

    p3.add_layout(area_label)
    p3.add_layout(length_label)

    def plot_source(coords):
        instance_linestring = LineString(coords[:, 0:2])
        instance_patch = instance_linestring.buffer(0.5)
        instance_x, instance_y = instance_patch.exterior.coords.xy
        return instance_x, instance_y

    def mapper(string, angle):
        angle = radians(angle)
        theta = 1.570 + angle

        num_chars = len(string)

        coords = np.zeros((num_chars + 1, 3), np.double)

        def makeRotMat(theta):
            rotMat = np.array((
                (cos(theta), -sin(theta), 0),
                (sin(theta), cos(theta), 0),
                (0, 0, 1)
            ))
            return rotMat

        rotVec = makeRotMat(theta)

        begin_vec = np.array((1, 0, 0), np.float64)
        i = 1

        for c in string:
            if c == 'F':
                next_vec = np.dot(rotVec, begin_vec)
                coords[i] = (
                    coords[i-1] + (1 * next_vec)
                )
                i += 1
                begin_vec = next_vec

            if c == '-':
                theta = theta - angle
                rotVec = makeRotMat(theta)

            if c == '+':
                theta = theta + angle
                rotVec = makeRotMat(theta)

        coords = np.delete(coords, np.s_[i:], 0)
        return coords

    def plot_creature(event):
        if len(source.selected.indices) > 0:

            creature_index = source.selected.indices[0]
            creature = plotData.iloc[creature_index, :]
            coords = np.array(ast.literal_eval(creature['Coordinates']))

            L_string.text = '{}'.format(creature['L-string'])
            area_label.text = 'Area: {:.2f}'.format(creature['Area'])
            length_label.text = 'Length of L-string: {}'.format(
                len(creature['L-string']))

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

            patch_x, patch_y = plot_source(coords)
            # x_points = [list(patch_x)]
            # y_points = [list(patch_y)]
            rules = ast.literal_eval(creature['Rules'])
            rules = rules['X']
            rules = rules['options']
            rule1_c = mapper(rules[0], creature['Angle'])
            rule2_c = mapper(rules[1], creature['Angle'])

            rule_text.text = rules[0] + '\n' + rules[1]

            rule1_morphology = LineString(rule1_c[:, 0:2])
            rule1_patch = rule1_morphology.buffer(0.5)
            rpatch_x, rpatch_y = rule1_patch.exterior.coords.xy

            rule2_morphology = LineString(rule2_c[:, 0:2])
            rule2_patch = rule2_morphology.buffer(0.5)
            r2patch_x, r2patch_y = rule2_patch.exterior.coords.xy

            inside_x = []
            inside_y = []
            # for i, _ in enumerate(creature_patch.interiors):
            #     x_in, y_in = creature_patch.interiors[i].coords.xy
            #     # x_points.append(list(x_in))
            #     # y_points.append(list(y_in))
            #     inside_x.append(x_in)
            #     inside_y.append(x_in)

            # x_points = [[x_points]]
            # y_points = [[y_points]]

            # source2.data = dict(x=x_points, y=y_points)

            source1.data = dict(x=coords[:, 0], y=coords[:, 1])
            source2.data = dict(x=patch_x, y=patch_y)
            source3.data = dict(x=inside_x, y=inside_y)
            rule1.data = dict(x=rpatch_x, y=rpatch_y)
            rule2.data = dict(x=r2patch_x, y=r2patch_y)
            p3.x_range = (min(coords), max(coords))
            p3.match_aspect = True

        else:
            source1.data = dict(x=[0, 0], y=[0, 0])
            source2.data = dict(x=[0, 0], y=[0, 0])
            source3.data = dict(x=[0, 0], y=[0, 0])
            rule1.data = dict(x=[0, 0], y=[0, 0])
            rule2.data = dict(x=[0, 0], y=[0, 0])
            L_string.text = 'Select creature'
            area_label.text = 'Select creature'
            length_label.text = 'Select creature'
            rule_text.text = 'Select creature'

    p1.on_event(Tap, plot_creature)
    p2.on_event(Tap, plot_creature)
    p4.on_event(Tap, plot_creature)

    a = row(L_string)
    b = row(p1, p2, p3)
    c_1 = column(p5, p6)
    c_2 = column(grams, rule_text)
    c = row(
        p4,
        Spacer(width=50),
        c_2,
        c_1)
    layout = column(a, b, c)

    doc.add_root(layout)


def main():
    """Launch the server and connect to it.
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
