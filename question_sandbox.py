import numpy as np
from shapely.geometry import MultiLineString, LineString
from shapely import ops, affinity
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from math import atan2, cos, sin, degrees
from descartes.patch import PolygonPatch
import time
import pandas as pd

Coords = np.array([
    [
        [0, 0, 'N', 0],
        [0, 1, 'N', 0],
        [-1, 1, 'Y', 20],
        [-1, 2, 'N', 0],
    ],
    [
        [0, 1, 'N', 0],
        [0.707, 1.707, 'Y', 90],
        [0.707, 2.707, 'Y', 30],
        [0.207, 3.573, 'Y', 45],
        [0.207, 4.573, 'N', 0],
    ],
], dtype=object)

fig, ax = plt.subplots()

patches = []
lines = []
Vs = []
r_lines = []

for line in Coords:
    line = np.asarray(line, dtype=object)
    num_coords = line[:, 0:2]
    cumm = 0

    indi_coords = []

    for i, joint in enumerate(line):

        if joint[2] == 'Y':

            """ --------------- BODY -------------------------------- """
            indi_coords.append((joint[0], joint[1]))
            if np.sign(joint[0]) < 0:
                ax.annotate(
                    'Revolve @' + str(joint[3]) + ' degrees',
                    xy=(joint[0], joint[1]),
                    xytext=(np.sign(joint[0])*50, 0),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="0.5",),
                    ha='right',
                )
            else:
                ax.annotate(
                    'Revolve @' + str(joint[3]) + ' degrees',
                    xy=(joint[0], joint[1]),
                    xytext=(np.sign(joint[0])*50, 0),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="0.5",),
                    ha='left',
                )

            new_coords = ((line[i+1][0]), (line[i+1][1]))
            angle = degrees(atan2(
                (new_coords[1] - joint[1]),
                (new_coords[0] - joint[0])
            ))
            """ --------------- PATCH -------------------------------- """
            lineA = LineString([(joint[0], joint[1]),
                                ((line[i+1][0]), (line[i+1][1]))])
            left_line = affinity.rotate(
                lineA, joint[3]/2, (joint[0], joint[1]))
            rigt_line = affinity.rotate(
                lineA, -joint[3]/2, (joint[0], joint[1]))

            Vs.append(MultiLineString([lineA, left_line, rigt_line]))

            """ --------------- ANGLE LINES -------------------------------- """
            if cumm > 0:
                rotate_angle = line[i-1][3]/2
                r_lines += [affinity.rotate(
                    Vs[-1],
                    j,
                    (line[i-1][0], line[i-1][1])
                ) for j in np.arange(-rotate_angle, rotate_angle+1, rotate_angle/3)
                ]

            cumm += 1

        else:
            indi_coords.append((joint[0], joint[1]))
            cumm = 0

    lines.append(np.asarray(indi_coords))

linestring = MultiLineString(lines)

creature = ops.unary_union([linestring] + r_lines + Vs)

c_patch = PolygonPatch(creature.buffer(0.5), fc='BLACK', alpha=0.1)
ax.add_patch(c_patch)

for line in linestring:
    x, y = line.xy
    ax.plot(x, y, 'r-')

for V in (Vs + r_lines):
    for line in V:
        x, y = line.xy
        ax.plot(x, y, 'b--', alpha=0.25)

ax.axis('equal')
plt.show()
