import numpy as np
# LinearRing, LineString, MultiLineString, Point, Polygon, box
from shapely.geometry import *
from shapely.ops import unary_union
from math import radians, cos, sin, pi
import re
import random
from descartes.patch import PolygonPatch
from matplotlib.patches import Circle, Wedge
from matplotlib import pyplot as plt
import matplotlib as mpl


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

        if params.get('angle') == 'random':
            self.Angle = radians(np.random.randint(0, 90))
        else:
            self.Angle = radians(params.get('angle'))

        self.recur(params.get('recurs'))
        self.Length = params.get('length')
        self.env = params.get('env')
        self.mapper()
        self.tolines()
        self.morphology()
        self.results()

    def recur(self, iters):
        for _ in range(iters+1):
            if self.L_string.count('X') == 0:
                break
            else:
                self.L_string = ''.join([self.next_char(c)
                                         for c in self.L_string])

        if self.Params.get('prune'):
            self.L_string = self.L_string[:500]

    def next_char(self, c):
        if c not in self.Rules:
            return c

        d = self.Rules[c]
        r = int(random.random() * len(d))
        self.Choices.append(r)

        return d[r+1]

        # rule = self.Rules.get(c, c)
        # if not rule == c:
        #     key, choice = random.choice(list(self.Rules.get(c).items()))
        #     self.Choices.append(key)
        #     return choice
        # else:
        #     return rule

    def mapper(self):
        """Converts L-string to coordinates

        Returns
        -------
        List
            List of coordinates
        """

        num_chars = len(self.L_string)

        coords = np.zeros((num_chars + 1, 4), np.double)
        nodes = np.zeros((1, 4), np.double)

        rotVec = np.array((
            (cos(self.Angle), -sin(self.Angle), 0),
            (sin(self.Angle), cos(self.Angle), 0),
            (0, 0, 1)
        ))

        start_vec = np.array((0, 1, 0), np.float64)
        curr_vec = start_vec
        i = 1

        for c in self.L_string:
            """
            1: Node
            2: Branch
            3: Saved
            """
            if c == 'F':
                coords[i, :3] = (coords[i-1, :3] + (self.Length * curr_vec))
                i += 1

            if c == '-':
                curr_vec = np.dot(curr_vec, (-1*rotVec))

            if c == '+':
                curr_vec = np.dot(curr_vec, rotVec)

            if c == '[':
                nodes = np.vstack((nodes, coords[i-1]))
                coords[i-1, 3] = 3
                nodes[-1, 3] = 1

            if c == ']':
                if coords[i-1, 3] == 1:
                    # coords[i, 3] = 2
                    coords[i-1] = nodes[-1]
                    # i += 1
                else:
                    coords[i-1, 3] = 2
                    if len(nodes) == 1:
                        coords[i] = nodes[-1]
                    else:
                        value, nodes = nodes[-1], nodes[:-1]
                        coords[i] = value
                    i += 1

        coords = np.delete(coords, np.s_[i:], 0)
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
            if (self.Coords[i, 3] == 2) or (i == (len(self.Coords) - 1)):
                lines.append(self.Coords[j:i+1, :2].tolist())
                j = i+1

        if not lines:
            self.Lines = [self.Coords[:, :3]]
        else:
            self.Lines = [line for line in lines if len(line) > 1]

    def morphology(self):
        """Converts coordinates or lines to shapely object and buffers

        Returns
        -------
        Shapely object
            L-creature "body"
        """

        if len(self.Lines) > 1:
            self.Linestring = MultiLineString(self.Lines)
        else:
            self.Linestring = LineString(self.Coords[:, :2])

        self.absorbA = self.Linestring.buffer(self.Length/2)

        self.Area = 0
        if not self.env.patches:
            self.Area = self.absorbA.area
        else:
            for patch in self.env.patches:
                p = PolygonPatch(patch)  # * patch._alpha)
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
                                                for s in re.findall(r'[' + char + ']+', self.L_string)))
            except:
                setattr(self, 'Max' + char, 0)

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
            width = self.richness
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
            for rad in radii[0]:
                coords = self.scale * 10 * \
                    np.array([np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1)])
                patch = Point(coords).buffer(rad)
                self.patches.append(patch)
