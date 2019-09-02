import numpy as np
from shapely.geometry import MultiLineString, LineString, Polygon
from shapely import ops, affinity
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection, LineCollection
from math import atan2, cos, sin, degrees
from descartes.patch import PolygonPatch
import time
import pandas as pd
from tabulate import tabulate
import traceback
from pathos.multiprocessing import ProcessPool
import multiprocessing as mp
import cProfile

import networkx as nx

import scipy.spatial as spatial

# def buffer_line(line):
#     return Polygon(line.buffer(0.5))

if __name__ == '__main__':
    Coords = np.array([
        [0, 0, 0, 0, 'N', 0, 0],
        [0, 1, 0, 'BRANCH', 'N', 0, 0],
        [0, 0, 0, 'BRANCH', 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-0.85, -0.51, 0, 'BRANCH', 'Y', 45, 0],
        [-0.85, -0.51, 0, 'NODE', 'Y', 45, 0],
        [-1.71, -1.03, 0, 0, 'Y', 45, 0],
        [-1.66, -2.02, 0, 'BRANCH', 'Y', 45, 0],
        [-1.66, -2.02, 0, 'NODE', 'Y', 45, 0],
        [-1.60, -3.02, 0, 'BRANCH', 'Y', 45, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0.90, -0.42, 0, 'BRANCH', 'Y', 45, 0],
        [0.90, -0.42, 0, 'NODE', 'Y', 45, 0],
        [1.81, -0.84, 0, 'BRANCH', 'Y', 45, 0],
        [0, 0, 0, 'BRANCH', 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0.10, -0.99, 0, 0, 'Y', 45, 0],
        [-0.69, -1.59, 0, 0, 'Y', 45, 0],
        [-0.53, -2.58, 0, 'BRANCH', 'Y', 45, 0],
        [-0.53, -2.58, 0, 'NODE', 'Y', 45, 0],
    ], dtype=object)

    for ind, coord in enumerate(Coords):
        if coord[3] == 'BRANCH':
            if (coord[0:3] == Coords[ind + 1, 0:3]).all():
                np.delete(Coords, ind, 0)

    lines = []

    j = 0
    for i in range(len(Coords)):
        if (Coords[i, 3] == 'BRANCH') or (i == (len(Coords) - 1)):
            lines.append(Coords[j:i+1].tolist())
            j = i+1

    if not lines:
        Lines = [Coords[:]]
    else:
        Lines = [line for line in lines if len(line) > 1]

    fig, ax = plt.subplots()

    patches = []
    lines = []
    Vs = []
    all_r_lines = []
    texts = []

    for num, line in enumerate(Lines):
        line = np.asarray(line, dtype=object)
        num_coords = line[:, 0:2]
        cumm = 0

        indi_coords = []

        for i, joint in enumerate(line):

            if joint[4] == 'Y' and joint[3] != 'BRANCH':

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
                    Lines[num][i][6] = cumm

                cumm += 1

            else:
                indi_coords.append((joint[0], joint[1]))
                cumm = 0

        lines.append(np.asarray(indi_coords))

    linestring = MultiLineString(lines)

    # print(tabulate(Lines))

    for num, line_coords in reversed(list(enumerate(Lines))):
        for i, joint in reversed(list(enumerate(line_coords))):

            if joint[4] == 'Y' and i < (len(Coords)-1) and joint[3] != 'BRANCH':

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
                        Vs.append(MultiLineString(
                            [lineA, left_line, rigt_line]))

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

                    all_r_lines = []

    # all_r_lines = [item for sublist in all_r_lines for item in sublist]

    all_lines = Vs

    a = ops.unary_union(all_lines)

    creature = (Vs + [linestring])

    # pieces = []
    # for l in creature:
    #     pieces.append(l)

    # with mp.Pool(2) as pool:
    #     result = list(pool.imap_unordered(buffer_line, pieces))

    # creature_poly = ops.unary_union(result)
    # absorbA = creature_poly
    # moves = Vs

    # polies = []
    # x = []
    # y = []
    # for l in creature:
    #     try:
    #         for a in l:
    #             x.append(a.xy[0])
    #             y.append(a.xy[1])
    #     except:
    #         x.append(l.xy[0])
    #         y.append(l.xy[1])
    #     polies.append(PolygonPatch(l.buffer(0.5)))

    # x = [item for sublist in x for item in sublist]
    # y = [item for sublist in y for item in sublist]

    # for c_l in linestring:
    #     x, y = c_l.xy
    #     ax.plot(x, y)
    # for p in polies:
    #     p.set_alpha(0.1)
    #     ax.add_artist(p)

    # ax.axis('equal')
    # plt.show()
    G = nx.Graph()

    # pos = {i: (a, b) for i, (a, b) in enumerate(zip(x, y))}
    # G.add_nodes_from(pos.keys())

    # XY = list(zip(x, y))
    # ax.scatter(x, y, c='red')

    # r = list(range(1, len(x)))
    # edges = [(c, d) for c, d in zip(XY, XY[1:])]

    nodes = {}
    i = 0

    for geom in creature:
        for line in geom:
            for seg_start, seg_end in zip(list(line.coords), list(line.coords)[1:]):
                G.add_edge(seg_start, seg_end)
                nodes[i] = (seg_start, seg_end)
                i += 1

    # nx.draw(G)
    print('\n')
    print(nodes[0][0])
    ax.plot(nodes[0][0][0], nodes[0][0][1], 'xr')
    print('\n')
    print(nodes[1][0])
    ax.plot(nodes[1][0][0], nodes[1][0][1], 'xr')
    print('\n')
    print(nx.shortest_path(G, nodes[0][0], nodes[1][0]))
    # plt.show()
    # creature_poly = ops.unary_union(polies)
    # creature_patch = PolygonPatch(creature_poly, fc='BLUE', alpha=0.1)

    # absorbA = creature_poly
    # moves = Vs

    # points = [p.get_verts() for p in polies]

    # all_points = np.row_stack(points)

    # in_all_points = np.all(
    #     [p.get_path().contains_points(all_points, p.get_transform(), radius=0.0) for p in polies], axis=0
    # )

    # transdata = ax.transData.inverted()
    # data_points = [transdata.transform(point) for point in points]
    # all_data_points = np.row_stack(data_points)
    # intersection = all_data_points[in_all_points]

    # hull = spatial.ConvexHull(intersection)
    # lc = LineCollection(
    #     intersection[hull.simplices], colors='black', linewidth=2, zorder=5)

    # ax.add_collection(lc)

    for c_l in linestring:
        x, y = c_l.xy
        ax.plot(x, y)

    for m in all_lines:
        for line in m:
            x, y = line.xy
            ax.plot(x, y, 'g--', alpha=0.25)

    # for patch in polies:
    #     ax.add_patch(PolygonPatch(patch, fc='BLUE', alpha=0.1))
    # ax.add_patch(creature_patch)
    # x, y = try_patch.get_xy()
    # ax[1].plot(x, y, 'r-')

    ax.axis('equal')
    plt.show()
