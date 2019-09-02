import numpy as np
# LinearRing, LineString, MultiLineString, Point, Polygon, box
from shapely.geometry import *
from shapely import ops, affinity
from math import radians, cos, sin, pi
import re
import random
import time
from descartes.patch import PolygonPatch
from matplotlib.patches import Circle, Wedge
from matplotlib import pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import traceback

from pathos.multiprocessing import ProcessPool


def buffer_line(line):
    # while True:
    #     line = in_queue.get()
    #     out_queue.put(Polygon(line.buffer(0.5)))
    return Polygon(line.buffer(0.5))


class Creature:
    """
    Generates a complete virtual creature
    Tests
    -----------

    """

    def __init__(self, params):
        """
        Initialises a simple L-system
        Parameters
        ----------
        variables : str
            a string containing all of the letters that take part in the
            recursion. These letters should also have associated rules.
        constants : str or None
            a string containing all the letters that do not take part in the
            recursion. These letters will not have an associated rule
        axiom : str
            The initial character string
        rules : dict
            a dictionary containing the rules for recursion. This is a
            dictionary of listing the letter replacement in the recursion.
            eg.
            {"A": "AB",
            "B": "A"}
        """
        self.Choices = []
        self.Params = params
        self.Rules = params.get('rules')
        self.L_string = params.get('axiom')
        self.Constants = params.get('constants')
        self.Variables = params.get('variables')
        self.Joints = params.get('joints')
        self.Joint_string = list(' ')

        if params.get('angle') == 'random':
            self.Angle = radians(np.random.randint(0, 90))
        else:
            self.Angle = radians(params.get('angle'))

        self.recur(params.get('recurs'))

        self.Joint_string = re.sub('[^YN]', '', self.Joint_string)

        self.Length = params.get('length')
        self.env = params.get('env')
        self.mapper()
        self.tolines()

        self.create_body()

        self.absorb_area()
        self.results()

    def recur(self, iters):
        for _ in range(iters+1):
            if self.L_string.count('X') == 0:
                break
            else:
                self.L_string = ''.join([self.next_char(c)
                                         for c in self.L_string])

        self.Joint_string = ''.join(
            [self.Joints['X'].get(i+1) for i in self.Choices])

        if self.Params.get('prune'):
            self.L_string = self.L_string[:500]

    def next_char(self, c):
        if c not in self.Rules:
            return c

        d = self.Rules[c]
        r = int(random.random() * len(d))
        self.Choices.append(r)

        return d[r+1]

    def mapper(self):
        """Converts L-string to coordinates

        Returns
        -------
        List
            List of coordinates
        """

        num_chars = len(self.L_string)

        coords = np.zeros((num_chars + 1, 7), dtype=object)
        nodes = np.zeros_like(coords)

        coords[0, 4:7] = ('N', 0, 0)

        rotVec = np.array((
            (cos(self.Angle), -sin(self.Angle), 0),
            (sin(self.Angle), cos(self.Angle), 0),
            (0, 0, 1)
        ))

        start_vec = np.array((0, 1, 0), np.float64)
        curr_vec = start_vec
        i = 1

        joints = list(self.Joint_string)

        for c in self.L_string:
            """
            1: Node
            2: Branch
            3: Saved
            """
            if c == 'F':
                if i == 1:
                    coords[i, 4:7] = ('N', 0, 0)
                else:
                    try:
                        coords[i, 4:7] = (joints.pop(0), 45, 0)
                    except:
                        pass

                coords[i, :3] = (coords[i-1, :3] + (self.Length * curr_vec))

                i += 1

            if c == '-':
                curr_vec = np.dot(curr_vec, (-1*rotVec))

            if c == '+':
                curr_vec = np.dot(curr_vec, rotVec)

            if c == '[':
                nodes = np.vstack((nodes, coords[i-1]))
                # coords[i-1, 3] = 3
                # nodes[-1, 3] = 1
                coords[i-1, 3] = 'SAVED'
                nodes[-1, 3] = 'NODE'

            if c == ']':
                if coords[i-1, 3] == 'NODE':  # coords[i-1, 3] == 1:
                    # coords[i, 3] = 2
                    coords[i-1] = nodes[-1]
                    # i += 1
                else:
                    # coords[i-1, 3] = 2
                    coords[i-1, 3] = 'BRANCH'
                    if len(nodes) == 1:
                        coords[i] = nodes[-1]
                    else:
                        value, nodes = nodes[-1], nodes[:-1]
                        coords[i] = value
                    i += 1

        coords = np.delete(coords, np.s_[i:], 0)

        for ind, line in enumerate(coords):
            if line[3] == 'BRANCH':
                if (line[0:3] == coords[ind + 1, 0:3]).all():
                    np.delete(coords, ind, 0)

        self.Coords = coords

    def tolines(self):
        """Converts L-string coordinates to individual line segments

        Returns
        -------
        List
            List of L-string lines
        """
        lines = []

        j = 0
        for i in range(len(self.Coords)):
            if (self.Coords[i, 3] == 'BRANCH') or (i == (len(self.Coords) - 1)):
                lines.append(self.Coords[j:i+1].tolist())
                j = i+1

        if not lines:
            self.Lines = [self.Coords[:]]
        else:
            self.Lines = [line for line in lines if len(line) > 1]

    def create_body(self):
        lines = []
        Vs = []
        all_r_lines = []

        for line in self.Lines:
            line = np.asarray(line, dtype=object)
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

                    # new_coords = ((line[i+1][0]), (line[i+1][1]))
                    # angle = degrees(atan2(
                    #     (new_coords[1] - joint[1]),
                    #     (new_coords[0] - joint[0])
                    # ))

                    if cumm > 0:
                        loc = np.where(np.all(joint == self.Coords, axis=1))
                        self.Coords[loc, 6] = cumm

                    cumm += 1

                else:
                    indi_coords.append((joint[0], joint[1]))
                    cumm = 0

            lines.append(np.asarray(indi_coords))

        if len(lines) > 1:
            self.Linestring = MultiLineString(lines)
        else:
            self.Linestring = LineString(self.Coords[:, :2])

        for i, joint in reversed(list(enumerate(self.Coords))):

            if joint[4] == 'Y' and i < (len(self.Coords)-1) and joint[3] != 'BRANCH':

                if joint[6] > 0:
                    """ --------------- PATCH -------------------------------- """
                    lineA = LineString([(joint[0], joint[1]),
                                        ((self.Coords[i+1][0]), (self.Coords[i+1][1]))])
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

                    rotate_angle = self.Coords[i-1][5]/2
                    r_lines = [affinity.rotate(
                        Vs[-1],
                        j,
                        (self.Coords[i-1][0], self.Coords[i-1][1])
                    ) for j in np.linspace(-rotate_angle, rotate_angle, num=3)
                    ]

                    all_r_lines += [r_lines]

                    Vs[-1] = ops.unary_union([Vs[-1]] + r_lines)

                else:
                    """ --------------- PATCH -------------------------------- """
                    lineA = LineString([(joint[0], joint[1]),
                                        ((self.Coords[i+1][0]), (self.Coords[i+1][1]))])
                    left_line = affinity.rotate(
                        lineA, joint[5]/2, (joint[0], joint[1]))
                    rigt_line = affinity.rotate(
                        lineA, -joint[5]/2, (joint[0], joint[1]))

                    Vs.append(MultiLineString(
                        [lineA, left_line, rigt_line]))

                    all_r_lines = []

        all_lines = Vs

        a = ops.unary_union(all_lines).simplify(0)

        creature = (Vs + [a] + [self.Linestring])

        polies = []
        pieces = []
        for l in creature:
            # polies.append(Polygon(l.buffer(0.5)))
            pieces.append(l)

        try:
            with ProcessPool(nodes=2) as pool:
                result = list(pool.uimap(buffer_line, pieces))
        except:
            traceback.print

        creature_poly = ops.unary_union(result)
        # creature_patch = PolygonPatch(creature_poly, fc='BLUE', alpha=0.1)

        self.absorbA = creature_poly

        # if isinstance(creature, MultiLineString):

        #     polies = []
        #     for l in creature:
        #         polies.append(Polygon(l.buffer(0.5)))

        #     # try:
        #     #     with ProcessPool(nodes=2) as pool:
        #     #         result = list(pool.uimap(buffer_line, pieces))
        #     # except:
        #     #     traceback.print_exc()

        #     creature_poly = ops.unary_union(polies)
        #     creature_patch = PolygonPatch(creature_poly, fc='BLUE', alpha=0.1)

        #     self.absorbA = creature_poly

        # else:
        #     self.absorbA = creature.buffer(self.Length/2)

        self.moves = Vs

        # fig, ax = plt.subplots(2, 1)

        # c_patch = PolygonPatch(creature.buffer(0.5), fc='BLACK', alpha=0.1)

        # ax.add_patch(c_patch)

        # try:
        #     for c_l in self.Linestring:
        #         x, y = c_l.xy
        #         ax[0].plot(x, y, 'r-')
        # except:
        #     x, y = self.Linestring.xy
        #     ax[0].plot(x, y, 'r-')

        # for m in all_lines:
        #     for line in m:
        #         x, y = line.xy
        #         ax[0].plot(x, y, 'g--', alpha=0.25)

        # ax[0].axis('equal')

        # try:
        #     for line in creature:
        #         x, y = line.xy
        #         ax[1].plot(x, y)
        # except:
        #     x, y = creature.xy
        #     ax[1].plot(x, y)

        # ax[1].axis('equal')
        # plt.show()

    def absorb_area(self):
        """Converts coordinates or lines to shapely object and buffers

        Returns
        -------
        Shapely object
            L-creature "body"
        """

        # if len(self.Lines) > 1:
        #     self.Linestring = MultiLineString(self.Lines)
        # else:
        #     self.Linestring = LineString(self.Coords[:, :2])

        # self.absorbA = self.Linestring.buffer(self.Length/2)

        self.Area = 0
        if not self.env.patches:
            self.Area = self.absorbA.area
        else:
            for patch in self.env.patches:
                # p = PolygonPatch(patch)  # * patch._alpha)
                self.Area += (self.absorbA.intersection(patch).area)

        self.Bounds = self.Linestring.bounds

    def results(self):
        chars = set(list(self.Constants + self.Variables))
        avgs = dict()

        for char in chars:
            avgs[char] = re.findall(
                '(?<=' + re.escape(char) + ').*?(?=' + re.escape(char) + ')', self.L_string)

        for char in chars:
            if len(avgs[char]) == 0:
                avgs[char] = 0.0
            else:
                avgs[char] = sum([len(char)
                                  for char in avgs[char]])/len(avgs[char])

        for char in chars:
            setattr(self, 'Percent' + char,
                    self.L_string.count(char)/len(self.L_string))
            setattr(self, 'Count' + char, self.L_string.count(char))
            setattr(self, 'Average' + char, avgs.get(char))

            try:
                setattr(self, 'Max' + char, max(len(s)
                                                for s in re.findall(r'\[' + char + '\]+', self.L_string)))
            except:
                setattr(self, 'Max' + char, 0)

        self.JointNo = self.Joint_string.count('Y')

        self.Comp = np.linalg.norm(self.Bounds)

        self.Fitness = self.Area/self.Comp

        self.Rules = list(self.Rules['X'].values())

        self.Efficiency = self.Area/self.absorbA.area

        self.Ratio = np.array([
            (self.Choices.count(1)/len(self.Choices)) *
            getattr(self, self.Params.get('fitness_metric')),
            (self.Choices.count(2)/len(self.Choices)) *
            getattr(self, self.Params.get('fitness_metric')),
        ])

        self.Generation = self.Params.get('Generation')


class Environment:
    """ Creates the environment
    Tests
    -----------

    """

    def __init__(self, params=None):
        scale = {
            'small': 4,
            'medium': 8,
            'large': 12,
        }

        richness = {
            'scarce': 1,
            'common': 5,
            'abundant': 10,
        }

        self.shape = params.get('shape')
        self.richness = richness[params.get('richness')]
        self.scale = scale[params.get('scale')]
        self.patches = []
        radius = (1 * self.scale)

        if self.shape == 'circle':
            # width = self.richness
            center = (0, 0)
            radius = (2 * self.scale)
            ring = Wedge(center, radius, 0, 360)
            ring_coords = ring._path.vertices
            ring_coords = ring_coords[(ring_coords[:, 0] != center[0]) & (
                ring_coords[:, 1] != center[1])]
            ring_patch = LinearRing(ring_coords).buffer(self.richness)
            self.patches.append(ring_patch)

        if self.shape == 'square':
            box_patch = LinearRing([
                (-radius, -radius),
                (-radius, radius),
                (radius, radius),
                (radius, -radius)]).buffer(self.richness)
            self.patches.append(box_patch)

        if self.shape == 'triangle':
            triangle_patch = LinearRing([
                (-0.866 * radius, -0.5 * radius),
                (0, radius),
                (0.866 * radius, -0.5 * radius)]).buffer(self.richness)
            self.patches.append(triangle_patch)

        if self.shape == 'rainbow':
            width = self.richness
            center = (0, 0)
            radius = (1 * self.scale)

            rainbow = Wedge(center, radius, 90, 270)

            # rotate = mpl.transforms.Affine2D().rotate_deg(random.randint(0, 360))

            # rainbow.set_transform(rotate)

            rainbow_coords = rainbow._path.vertices
            rainbow_patch = LineString(
                rainbow_coords[:13]).buffer(self.richness)
            self.patches.append(rainbow_patch)

        if self.shape == 'patches':
            radii = radius * \
                np.random.dirichlet(np.ones(self.richness), 1)

            coords = np.random.random((len(radii[0]), 2))

            for i in np.random.randint(0, len(coords), 4):
                coords[i] = coords[i] * -1

            for rad in radii[0]:
                coords = self.scale * \
                    np.array([np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1)])
                patch = Point(coords).buffer(rad)
                self.patches.append(patch)
