# import numpy as np

# import vispy.plot as vp

# n = 100000
# data = np.random.randn(n, 2)
# color = (0.8, 0.25, 0)
# n_bins = 100

# fig = vp.Fig(show=False)
# fig[0:4, 0:4].plot(data, symbol='o', width=0, face_color=color + (0.05,), edge_color=None,
#                    marker_size=10)
# fig[4, 0:4].histogram(data[:, 0], bins=n_bins, color=color, orientation='h')
# fig[0:4, 4].histogram(data[:, 1], bins=n_bins, color=color, orientation='v')

# if __name__ == '__main__':
#     fig.show(run=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# g = sns.JointGrid("total_bill", "tip", tips)
# for day, day_tips in tips.groupby("day"):
#     sns.kdeplot(day_tips["total_bill"], ax=g.ax_marg_x, legend=False)
#     sns.kdeplot(day_tips["tip"], ax=g.ax_marg_y, vertical=True, legend=False)
#     g.ax_joint.plot(day_tips["total_bill"], day_tips["tip"], "o", ms=5)

# plt.show()

from math import sin
from random import random

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform

list_x = list(range(100))
list_y = [random() + sin(i / 20) for i in range(100)]
desc = [str(i) for i in list_y]

source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=desc))
hover = HoverTool(tooltips=[
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ('desc', '@desc'),
])
mapper = LinearColorMapper(palette=plasma(
    256), low=min(list_y), high=max(list_y))

p = figure(plot_width=400, plot_height=400,
           tools=[hover], title="Belgian test")
p.circle('x', 'y', size=10, source=source,
         fill_color=transform('y', mapper))

output_file('test.html')
show(p)
