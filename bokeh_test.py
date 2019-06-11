# from random import random

# from bokeh.layouts import column
# from bokeh.models import Button
# from bokeh.palettes import RdYlBu3
# from bokeh.plotting import figure, curdoc

# # create a plot and style its properties
# p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
# p.border_fill_color = 'black'
# p.background_fill_color = 'black'
# p.outline_line_color = None
# p.grid.grid_line_color = None

# # add a text renderer to our plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
#            text_baseline="middle", text_align="center")

# i = 0

# ds = r.data_source

# # create a callback that will add a number in a random location


# def callback():
#     global i

#     # BEST PRACTICE --- update .data in one step with a new dict
#     new_data = dict()
#     new_data['x'] = ds.data['x'] + [random()*70 + 15]
#     new_data['y'] = ds.data['y'] + [random()*70 + 15]
#     new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i % 3]]
#     new_data['text'] = ds.data['text'] + [str(i)]
#     ds.data = new_data

#     i = i + 1


# # add a button widget and configure with the call back
# button = Button(label="Press Me")
# button.on_click(callback)

# # put the button and plot in a layout and add to the document
# curdoc().add_root(column(button, p))

# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, Column
# from bokeh.io import curdoc
# from bokeh.events import DoubleTap

# coordList = []

# TOOLS = "tap"
# bound = 10
# p = figure(title='Double click to leave a dot.',
#            tools=TOOLS, width=700, height=700,
#            x_range=(-bound, bound), y_range=(-bound, bound))

# source = ColumnDataSource(data=dict(x=[], y=[]))
# p.circle(source=source, x='x', y='y')

# # add a dot where the click happened


# def callback(event):
#     Coords = (event.x, event.y)
#     coordList.append(Coords)
#     source.data = dict(x=[i[0] for i in coordList], y=[i[1]
#                                                        for i in coordList])


# p.on_event(DoubleTap, callback)

# layout = Column(p)

# curdoc().add_root(layout)

from tornado.ioloop import IOLoop

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
import os
import pandas as pd


def modify_doc(doc):
    """Add a plotted function to the document.

    Arguments:
        doc: A bokeh document to which elements can be added.
    """
    # x_values = range(10)
    # y_values = [x ** 2 for x in x_values]
    data_source = ColumnDataSource(data=dict(x=x_values, y=y_values))
    plot = figure(title="f(x) = x^2",
                  tools="crosshair,pan,reset,save,wheel_zoom",)
    plot.line('x', 'y', source=data_source, line_width=3, line_alpha=0.6)
    doc.add_root(plot)
    doc.title = "Test Plot"


def main():
    curDir = os.path.dirname(__file__)
    plotData = pd.read_csv(os.path.join(curDir, 'test.csv'))
    """Launch the server and connect to it.
    """
    global x_values, y_values
    x_values = plotData['Area'].values
    y_values = plotData['% of F'].values

    print("Preparing a bokeh application.")
    io_loop = IOLoop.current()
    bokeh_app = Application(FunctionHandler(
        modify_doc))

    server = Server({"/": bokeh_app}, io_loop=io_loop)
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")

    io_loop.add_callback(server.show, "/")
    io_loop.start()


main()
