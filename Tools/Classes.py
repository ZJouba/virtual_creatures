import numpy as np
from shapely.geometry import LineString, MultiLineString
from math import radians, cos, sin, pi
import re
import random


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
        self.Angle = radians(params.get('angle'))
        self.recur(params.get('recurs'))
        self.Length = params.get('length')
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
        rule = self.Rules.get(c, c)
        if not rule == c:
            key, choice = random.choice(list(self.Rules.get(c).items()))
            self.Choices.append(key)
            return choice
        else:
            return rule

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
                    coords[i, 3] = 2
                    coords[i-1] = nodes[-1]
                    i += 1
                else:
                    coords[i-1, 3] = 2
                    coords[i] = nodes[-1]
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

        self.Area = self.Linestring.buffer(self.Length/2).area
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

        self.Ratio = np.array([
            (self.Choices.count(1)/len(self.Choices)) * self.Fitness,
            (self.Choices.count(2)/len(self.Choices)) * self.Fitness,
        ])

        self.Rules = list(self.Rules['X'].values())

        self.Efficiency = self.Area/self.CountF


class Environment:
    """ Creates the environment
    Tests
    -----------

    """

    def __init__(self, params):

        self.params = params

        self.growth_coords = []
        for _ in range(3):
            self.growth_coords.append([
                random.randint(-30, 30), random.randint(-30, 30)
            ])
        self.patches = []
        self.circles = []
        for coord in self.growth_coords:
            self.patches.append(matplotlib.patches.Circle(
                coord, radius=random.randint(5, 10), color='red', alpha=np.random.uniform(0, 1)))
            self.circles.append(Point(coord).buffer(5))
