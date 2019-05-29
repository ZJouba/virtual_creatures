import pandas as pd
import time
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import itertools
import numpy as np
from tkinter import filedialog
import tkinter as tk
import os
import seaborn as sns
import matplotlib.pyplot as plt
from grid_strategy import strategies
import gc

# from bokeh.layouts import gridplot
# from bokeh.plotting import figure, show, output_file
# from bokeh.sampledata.autompg import autompg
# from bokeh.transform import jitter
# from bokeh.models import (BasicTicker, ColumnDataSource, Grid, LinearAxis,
#                           DataRange1d, PanTool, Plot, WheelZoomTool)
# from bokeh.models.glyphs import Circle

root = tk.Tk()
root.withdraw()
curDir = os.path.dirname(__file__)

ProgressBar().register()

filepath = filedialog.askopenfilename()
filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]

monteData = dd.read_csv(filepath)

monteData.fillna(0)

plotData = monteData.sample(frac=0.01)

plotData = plotData.compute()

gc.collect()

plotData['L-string'].drop_duplicates()
plotData.reset_index()

gc.collect()

plotData = plotData[['Area', '% of F', '% of +', '% of -', 'Longest F sequence', 'Longest + sequence',
                     'Longest - sequence', 'Average chars between Fs', 'Average chars between +s', 'Average chars between -s']]

num_render = len(plotData.columns)
specs = strategies.SquareStrategy('center').get_grid(num_render)

fig = plt.figure()

gc.collect()

for column, sub in zip(plotData.columns, specs):
    ax = fig.add_subplot(sub)
    if column in ['Longest F sequence', 'Longest + sequence', 'Longest - sequence', 'Average chars between Fs', 'Average chars between +s', 'Average chars between -s']:
        sns.barplot(plotData.Area, plotData[column], ax=ax)
    else:
        # sns.scatterplot(plotData.Area, plotData[column], ax=ax, size=0.5)
        sns.jointplot(plotData.Area, plotData[column], kind='kde')


# xdr = DataRange1d(bounds=None)
# ydr = DataRange1d(bounds=None)


# def make_plot(xname, yname, type):
#     plot = Plot(
#         x_range=xdr, y_range=ydr, background_fill_color="#efe8e2",
#         border_fill_color='white', plot_width=200, plot_height=200,
#         min_border_left=2, min_border_right=2, min_border_top=2, min_border_bottom=2)

#     if type == 'Bar':
#         plot.vbar(x=xname, y=yname)
#     elif type == 'Circle':
#         circle = Circle(x=xname, y=yname, fill_color="color",
#                         fill_alpha=0.2, size=4, line_color="color")
#         r = plot.add_glyph(source, circle)
#         xdr.renderers.append(r)
#         ydr.renderers.append(r)

#     xticker = BasicTicker()
#     plot.add_layout(Grid(dimension=0, ticker=xticker))

#     yticker = BasicTicker()
#     plot.add_layout(Grid(dimension=1, ticker=yticker))

#     plot.add_tools(PanTool(), WheelZoomTool())

#     return plot


# plots = []
# for column in plotData.columns:
#     if column in ['Longest F sequence', 'Longest + sequence', 'Longest - sequence', 'Average chars between Fs', 'Average chars between +s', 'Average chars between -s']:
#         window = make_plot(plotData['Area'], plotData[column], type='Bar')
#         plots.append(window)
#     else:
#         window = make_plot(plotData['Area'], plotData[column], type='Circle')
#         plots.append(window)


# show(gridplot(plots))
plt.show()
