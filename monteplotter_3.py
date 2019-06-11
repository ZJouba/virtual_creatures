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
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import RdYlGn10 as palette
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
import itertools


def modify_doc(doc):
    curDir = os.path.dirname(__file__)
    root = tk.Tk()
    root.withdraw()

    ProgressBar().register()

    filepath = filedialog.askopenfilename()
    filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]

    monteData = dd.read_csv(filepath)

    monteData.fillna(0)

    plotData = monteData.compute()

    gc.collect()

    plotData.drop_duplicates(subset='L-string', inplace=True)
    plotData.reset_index()
    for i in range(2, 6):
        plotData['{}-gram'.format(i)] = plotData['L-string'].apply(lambda x: [x[j:j+i]
                                                                              for j in range(0, len(x), i)])

    gc.collect()

    scatter = ColumnDataSource(data=plotData)
    line = ColumnDataSource(data=dict(x=[0, 0], y=[0, 0]))
    rule1 = ColumnDataSource(
        data=dict(x=[0, 0], y=[0, 0]))
    rule2 = ColumnDataSource(
        data=dict(x=[0, 0], y=[0, 0]))
    polygon = ColumnDataSource(data=dict(x=[0], y=[0]))

    rule1_poly = ColumnDataSource(
        data=dict(x=[0, 0], y=[0, 0]))
    rule2_poly = ColumnDataSource(
        data=dict(x=[0, 0], y=[0, 0]))

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

    plots_width = 500
    plots_height = 500
    p1 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl", tooltips=tooltips1)
    p1.xaxis.axis_label = 'Area'
    p1.yaxis.axis_label = '% of character'
    p1.scatter('Area', '% of F', size=7,
               source=scatter, color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    p2 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl", tooltips=tooltips2)
    p2.xaxis.axis_label = 'Area'
    p2.yaxis.axis_label = 'Length of sequence'
    p2.scatter('Area', 'Longest F sequence', size=7, source=scatter,
               fill_color='red', color=mapper, alpha=0.6, nonselection_fill_color=mapper)

    p3 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Selected Creature", output_backend="webgl")
    p3.axis.visible = False
    p3.grid.visible = False
    p3.line(x='x', y='y', line_color='red', source=line)
    p3.multi_polygons(xs='x', ys='y', source=polygon)

    p4 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
                title="Area", output_backend="webgl")
    p4.scatter('Area', 'Angle', size=7,
               source=scatter, color=mapper, alpha=0.6, nonselection_fill_color=mapper)
    p4.xaxis.axis_label = 'Area'
    p4.yaxis.axis_label = 'Angle (degrees)'

    p5 = figure(plot_width=plots_width, plot_height=plots_height//2,
                title="Rule 1", output_backend="webgl")
    p5.line(x='x', y='y', line_color='red', source=rule1)
    p5.multi_polygons(xs='x', ys='y', source=rule1_poly)
    p5.axis.visible = False
    p5.grid.visible = False

    p6 = figure(plot_width=plots_width, plot_height=plots_height//2,
                title="Rule 2", output_backend="webgl")
    p6.line(x='x', y='y', line_color='red', source=rule2)
    p6.multi_polygons(xs='x', ys='y', source=rule2_poly)
    p6.axis.visible = False
    p6.grid.visible = False

    L_string = Paragraph(text='Select creature', width=1500)

    grams = PreText(text='Select creature', width=400)
    rule_text = PreText(text='Select creature', width=400)

    area_label = Label(
        x=0,
        y=450,
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
        y=420,
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
        theta = 0

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

        dir_vec = np.array((0, 1, 0), np.float64)
        i = 1

        for c in string:
            if c == 'F':
                coords[i] = (
                    coords[i-1] + (1 * dir_vec)
                )
                i += 1

            if c == '-':
                theta = theta - angle
                rotVec = makeRotMat(theta)
                dir_vec = np.dot(rotVec, dir_vec)

            if c == '+':
                theta = theta + angle
                rotVec = makeRotMat(theta)
                dir_vec = np.dot(rotVec, dir_vec)

        coords = np.delete(coords, np.s_[i:], 0)
        return coords

    def plot_creature(event):

        if len(scatter.selected.indices) > 0:

            creature_index = scatter.selected.indices[0]
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

            p3.match_aspect = True

            rules = ast.literal_eval(creature['Rules'])
            rules = rules['X']
            rules = rules['options']

            rule_text.text = 'Rule 1: \t' + \
                rules[0] + '\n' + 'Rule 2: \t' + rules[1]

            if any(char == 'F' for string in rules[0] for char in string):
                rule1_c = mapper(rules[0], creature['Angle'])

                rule1_morphology = LineString(rule1_c[:, 0:2])
                rule1_patch = rule1_morphology.buffer(0.5)
                rpatch_x, rpatch_y = rule1_patch.exterior.coords.xy

                r1_points_x = [list(rpatch_x)]
                r1_points_y = [list(rpatch_y)]

                for i, _ in enumerate(rule1_patch.interiors):
                    x_in, y_in = creature_patch.interiors[i].coords.xy
                    r1_points_x.append(list(x_in))
                    r1_points_y.append(list(y_in))

                r1_points_x = [[r1_points_x]]
                r1_points_y = [[r1_points_y]]

                rule1.data = dict(
                    x=rule1_morphology.coords.xy[0], y=rule1_morphology.coords.xy[1])
                rule1_poly.data = dict(x=r1_points_x, y=r1_points_y)

                p5.match_aspect = True

            if any(char == 'F' for string in rules[1] for char in string):
                rule2_c = mapper(rules[1], creature['Angle'])

                rule2_morphology = LineString(rule2_c[:, 0:2])
                rule2_patch = rule2_morphology.buffer(0.5)
                r2patch_x, r2patch_y = rule2_patch.exterior.coords.xy

                r2_points_x = [list(r2patch_x)]
                r2_points_y = [list(r2patch_y)]

                for i, _ in enumerate(rule2_patch.interiors):
                    x_in, y_in = creature_patch.interiors[i].coords.xy
                    r2_points_x.append(list(x_in))
                    r2_points_y.append(list(y_in))

                r2_points_x = [[r2_points_x]]
                r2_points_y = [[r2_points_y]]

                rule2.data = dict(
                    x=rule2_morphology.coords.xy[0], y=rule2_morphology.coords.xy[1])
                rule2_poly.data = dict(x=r2_points_x, y=r2_points_y)

                p6.match_aspect = True

        else:
            line.data = dict(x=[0, 0], y=[0, 0])
            polygon.data = dict(x=[0, 0], y=[0, 0])
            rule1.data = dict(x=[0, 0], y=[0, 0])
            rule2.data = dict(x=[0, 0], y=[0, 0])
            rule1_poly.data = dict(x=[0, 0], y=[0, 0])
            rule2_poly.data = dict(x=[0, 0], y=[0, 0])
            L_string.text = 'Select creature'
            area_label.text = 'Select creature'
            length_label.text = 'Select creature'
            rule_text.text = 'Select creature'

    p1.on_event(Tap, plot_creature)
    p2.on_event(Tap, plot_creature)
    p4.on_event(Tap, plot_creature)

    top_row = row(L_string)
    middle_row = row(p1, p2, p4)
    bottom_row_right = column(p5, p6)
    bottom_row_middle = column(grams, rule_text)
    bottom_row = row(
        p3,
        Spacer(width=50),
        bottom_row_middle,
        Spacer(width=50),
        bottom_row_right)
    layout = column(
        top_row,
        middle_row,
        bottom_row)

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
