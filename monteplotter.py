from bokeh.io import output_file, show, export_png, export_svgs
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, Label, BoxZoomTool, ResetTool
from bokeh.layouts import gridplot
from bokeh.models.glyphs import Patch
from bokeh.events import Tap
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

curDir = os.path.dirname(__file__)
root = tk.Tk()
root.withdraw()

ProgressBar().register()

filepath = filedialog.askopenfilename()
filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]

monteData = dd.read_csv(filepath)

monteData.fillna(0)

plotData = monteData.sample(frac=0.0001)

plotData = plotData.compute()

gc.collect()

# plotData = pd.read_csv(os.path.join(curDir, 'test.csv'))

plotData['L-string'].drop_duplicates()
plotData.reset_index()

gc.collect()

plotData = plotData[['L-string', 'Area', '% of F', '% of +', '% of -', 'Longest F sequence', 'Longest + sequence',
                     'Longest - sequence', 'Average chars between Fs', 'Average chars between +s', 'Average chars between -s']]

source = ColumnDataSource(data=plotData)

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
p1.scatter('Area', '% of F', size=7, source=source,
           fill_color='red', legend='F', alpha=0.6)

p2 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
            title="Area", output_backend="webgl", tooltips=tooltips2)
p2.xaxis.axis_label = 'Area'
p2.yaxis.axis_label = 'Length of sequence'
p2.scatter('Area', 'Longest F sequence', size=7, source=source,
           fill_color='red', legend='F', alpha=0.6)
p2.scatter('Area', 'Longest + sequence', size=7, source=source,
           fill_color='blue', legend='+', alpha=0.6)
p2.scatter('Area', 'Longest - sequence', size=7, source=source,
           fill_color='green', legend='-', alpha=0.6)

p3 = figure(plot_width=plots_width, plot_height=plots_height, tools='pan,wheel_zoom,box_zoom,reset,tap,save',
            title="Selected Creature", output_backend="webgl", toolbar_location="below")
# p3.line(x=coords[:, 0], y=coords[:, 1], line_color='red')
# p3.patch(x=patch_x, y=patch_y)
# for i, _ in enumerate(creature_patch.interiors):
#     inside_x, inside_y = creature_patch.interiors[i].coords.xy
#     p3.patch(x=inside_x, y=inside_y, fill_color='white')

g = gridplot([[p1, p2, p3]])


def plot_creature(event):
    creature_index = source.selected.index
    print(source.selected.index)
    creature = plotData.loc[creature_index, :]

    coords = np.array(ast.literal_eval(creature['Coordinates']))

    creature_morphology = LineString(coords[:, 0:2])
    creature_patch = creature_morphology.buffer(0.5)
    patch_x, patch_y = creature_patch.exterior.coords.xy

    p3.line(x=coords[:, 0], y=coords[:, 1], line_color='red')
    p3.patch(x=patch_x, y=patch_y)
    for i, _ in enumerate(creature_patch.interiors):
        inside_x, inside_y = creature_patch.interiors[i].coords.xy
        p3.patch(x=inside_x, y=inside_y, fill_color='white')

    area_label = Label(
        x=0,
        y=400,
        x_units='screen',
        y_units='screen',
        text='Area: {:.2f}'.format(creature['Area']),
        render_mode='css',
        border_line_color='black',
        border_line_alpha=1.0,
        background_fill_color='white',
        background_fill_alpha=1.0,
    )

    length_label = Label(
        x=0,
        y=380,
        x_units='screen',
        y_units='screen',
        text='Length of L-string: {}'.format(len(creature['L-string'])),
        render_mode='css',
        border_line_color='black',
        border_line_alpha=1.0,
        background_fill_color='white',
        background_fill_alpha=1.0,
    )

    p3.add_layout(area_label)
    p3.add_layout(length_label)


p1.on_event(Tap, plot_creature)
p2.on_event(Tap, plot_creature)
# export_png(g, filename=os.path.join(curDir, "plot.png"))

curdoc().add_root(g)
