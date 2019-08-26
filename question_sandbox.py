import numpy as np
from shapely.geometry import MultiLineString, LineString, Polygon
from shapely import ops, affinity
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from math import atan2, cos, sin, degrees
from descartes.patch import PolygonPatch
import time
import pandas as pd
from adjustText import adjust_text

Coords = np.array([[
    [0, 0, 0, 0, 'N', 0, 0],
    [0, 1, 0, 0, 'N', 0, 0],
    [-0.99452, 0.895472, 0, 0, 'Y', 45, 0],
    [-1.98904, 0.790943, 0, 3, 'Y', 45, 0],
    [-2.97385, 0.617295, 0, 0, 'N', 0, 0],
    [-2.76594, -0.360853, 0, 0, 'N', 0, 0],
    [-1.79564, -0.118931, 0, 0, 'N', 0, 0],
]], dtype=object)

lines = []

# j = 0
# for i in range(len(Coords)):
#     if (Coords[i, 3] == 2) or (i == (len(Coords) - 1)):
#         lines.append(Coords[j:i+1].tolist())
#         j = i+1

# if not lines:
#     Lines = [Coords[:]]
# else:
#     Lines = [line for line in lines if len(line) > 1]

fig, ax = plt.subplots(2, 1)

patches = []
lines = []
Vs = []
all_r_lines = []
texts = []

for num, line in enumerate(Coords):
    line = np.asarray(line, dtype=object)
    num_coords = line[:, 0:2]
    cumm = 0

    indi_coords = []

    for i, joint in enumerate(line):

        if joint[4] == 'Y':

            """ --------------- BODY -------------------------------- """
            indi_coords.append((joint[0], joint[1]))
            # if np.sign(joint[0]) < 0:
            #     ax.annotate(
            #         'Revolve @ ' + str(joint[3]) + ' degrees',
            #         xy=(joint[0], joint[1]),
            #         xytext=(np.sign(joint[0])*200, 0),
            #         textcoords="offset points",
            #         arrowprops=dict(arrowstyle="->", color="0.5",),
            #         ha='right',
            #     )
            # else:
            #     ax.annotate(
            #         'Revolve @ ' + str(joint[3]) + ' degrees',
            #         xy=(joint[0], joint[1]),
            #         xytext=(np.sign(joint[0])*200, 0),
            #         textcoords="offset points",
            #         arrowprops=dict(arrowstyle="->", color="0.5",),
            #         ha='left',
            #     )

            new_coords = ((line[i+1][0]), (line[i+1][1]))
            angle = degrees(atan2(
                (new_coords[1] - joint[1]),
                (new_coords[0] - joint[0])
            ))

            if cumm > 0:
                Coords[num][i][6] = cumm

            cumm += 1

        else:
            indi_coords.append((joint[0], joint[1]))
            cumm = 0

    lines.append(np.asarray(indi_coords))

linestring = MultiLineString(lines)

for num, line_coords in reversed(list(enumerate(Coords))):
    for i, joint in reversed(list(enumerate(line_coords))):

        if joint[4] == 'Y':

            if joint[6] > 0:
                """ --------------- PATCH -------------------------------- """
                lineA = LineString([(joint[0], joint[1]),
                                    ((line_coords[i+1][0]), (line_coords[i+1][1]))])
                left_line = affinity.rotate(
                    lineA, joint[5]/2, (joint[0], joint[1]))
                rigt_line = affinity.rotate(
                    lineA, -joint[5]/2, (joint[0], joint[1]))

                try:
                    Vs[-1] = ops.unary_union([MultiLineString(
                        [lineA, left_line, rigt_line])] + all_r_lines[-1])
                except:
                    Vs.append(MultiLineString([lineA, left_line, rigt_line]))

                """ --------------- ANGLE LINES -------------------------------- """

                rotate_angle = line_coords[i-1][5]/2
                r_lines = [affinity.rotate(
                    Vs[-1],
                    j,
                    (line_coords[i-1][0], line_coords[i-1][1])
                ) for j in np.linspace(-rotate_angle, rotate_angle, num=3)
                ]

                all_r_lines += [r_lines]

                Vs[-1] = ops.unary_union([Vs[-1]] + r_lines)

            else:
                """ --------------- PATCH -------------------------------- """
                lineA = LineString([(joint[0], joint[1]),
                                    ((line_coords[i+1][0]), (line_coords[i+1][1]))])
                left_line = affinity.rotate(
                    lineA, joint[5]/2, (joint[0], joint[1]))
                rigt_line = affinity.rotate(
                    lineA, -joint[5]/2, (joint[0], joint[1]))

                Vs.append(MultiLineString([lineA, left_line, rigt_line]))

all_r_lines = [item for sublist in all_r_lines for item in sublist]

all_lines = Vs

a = ops.unary_union(all_lines)

creature = ops.unary_union([a] + [linestring])

start = time.time()
try_list = []
for line in creature:
    try_list.append(Polygon(line.buffer(0.5)))
try_poly = ops.cascaded_union(try_list)
try_patch = PolygonPatch(try_poly, fc='BLUE', alpha=0.1)
end = time.time()

print(end - start)

start = time.time()
c_patch = PolygonPatch(creature.buffer(0.5), fc='BLACK', alpha=0.1)
end = time.time()

print(end - start)

ax[0].add_patch(c_patch)

for c_l in linestring:
    x, y = c_l.xy
    ax[0].plot(x, y, 'r-')

for m in all_lines:
    for line in m:
        x, y = line.xy
        ax[0].plot(x, y, 'g--', alpha=0.25)

ax[0].axis('equal')

ax[1].add_patch(try_patch)
# x, y = try_patch.get_xy()
# ax[1].plot(x, y, 'r-')

ax[1].axis('equal')
plt.show()
