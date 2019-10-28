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
        if "[" in self.Constants:
            self.tolines()
        else:
            self.Lines = []
        self.layout()
        self.results()

    def recur(self, iters):
        for _ in range(iters+1):
            if self.L_string.count('X') == 0:
                break
            else:
                self.L_string = ''.join([self.next_char(c)
                                         for c in self.L_string])

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

        coords = np.zeros((num_chars + 1, 4), dtype=object)
        nodes = np.zeros_like(coords)

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
                coords[i-1, 3] = 'SAVED'
                nodes[-1, 3] = 'NODE'

            if c == ']':
                if coords[i-1, 3] == 'NODE':
                    coords[i-1] = nodes[-1]
                else:
                    coords[i-1, 3] = 'BRANCH'
                    if len(nodes) == 1:
                        coords[i] = nodes[-1]
                    else:
                        value, nodes = nodes[-1], nodes[:-1]
                        coords[i] = value
                    i += 1

            if c == '_':
                break

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
                lines.append(self.Coords[j:i+1, :2].tolist())
                j = i+1

        if not lines:
            self.Lines = [self.Coords[:]]
        else:
            self.Lines = [line for line in lines if len(line) > 1]

    def layout(self):
        """Converts coordinates or lines to shapely object and buffers

        Returns
        -------
        Shapely object
            L-creature "body"
        """

        if len(self.Lines) > 1:
            self.Linestring = MultiLineString(self.Lines)
            overall_len = self.Linestring.length
            max_len = max([line.length for line in self.Linestring]
                          )/overall_len

        else:
            self.Linestring = LineString(self.Coords[:, :2])
            max_len = 1

        self.Area = self.Linestring.buffer(self.Length/2).area
        self.Bounds = self.Linestring.bounds
        self.Penalty = self.Area * max_len

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

        self.Comp = np.linalg.norm(self.Bounds)

        self.Fitness = self.Area/self.Comp

        self.Rules = list(self.Rules['X'].values())

        self.Efficiency = self.Area

        self.Ratio = np.array([
            (self.Choices.count(1)/len(self.Choices)) *
            getattr(self, self.Params.get('fitness_metric')),
            (self.Choices.count(2)/len(self.Choices)) *
            getattr(self, self.Params.get('fitness_metric')),
        ])

        self.Generation = self.Params.get('Generation')

        self.absorbA = self.Area


class B_Creature:
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
        self.choices = []
        self.rules = params.get('rules')
        self.l_string = params.get('axiom')
        self.constants = params.get('constants')
        self.variables = params.get('variables')
        self.angle = radians(params.get('angle'))
        self.recur(params.get('recurs'))
        self.length = params.get('length')
        self.mapper()
        self.tolines()
        self.layout()
        self.results()

    # def recur(self, n):
    #     for _ in range(n):
    #         self.l_string = ''.join([self.next_char(c) for c in self.l_string])
    #     self.l_string = self.l_string.replace('X', '')

    def recur(self, iters):
        for _ in range(iters+1):
            if self.l_string.count('X') == 0:
                break
            else:
                self.l_string = ''.join([self.next_char(c)
                                         for c in self.l_string])

    def next_char(self, c):
        rule = self.rules.get(c, c)
        if not rule == c:
            key, choice = random.choice(list(self.rules.get(c).items()))
            self.choices.append(key)
            return choice
        else:
            return rule
        # rule = self.rules.get(c, c)
        # if len(rule) > 1:
        #     return np.random.choice(self.rules.get(c, ["Other"])["options"],
        #                             p=self.rules.get(c, ["Other"])[
        #                                 "probabilities"])
        # else:
        #     return rule

    def mapper(self):
        """Converts L-string to coordinates

        Returns
        -------
        List
            List of coordinates
        """

        num_chars = len(self.l_string)

        coords = np.zeros((num_chars + 1, 4), np.double)
        nodes = np.zeros((1, 4), np.double)

        rotVec = np.array((
            (cos(self.angle), -sin(self.angle), 0),
            (sin(self.angle), cos(self.angle), 0),
            (0, 0, 1)
        ))

        start_vec = np.array((0, 1, 0), np.float64)
        curr_vec = start_vec
        i = 1

        for c in self.l_string:
            if c == 'F':
                coords[i, :3] = (coords[i-1, :3] + (self.length * curr_vec))
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
        self.coords = coords

    def tolines(self):
        """Converts L-string coordinates to individual line segments

        Returns
        -------
        List
            List of L-string lines
        """
        lines = []

        j = 0
        for i in range(len(self.coords)):
            if (self.coords[i, 3] == 2) or (i == (len(self.coords) - 1)):
                lines.append(self.coords[j:i+1, :2].tolist())
                j = i+1

        if not lines:
            self.lines = [self.coords[:, :3]]
        else:
            self.lines = [line for line in lines if len(line) > 1]

    def layout(self):
        """Converts coordinates or lines to shapely object and buffers

        Returns
        -------
        Shapely object
            L-creature "body"
        """

        if len(self.lines) > 1:
            self.linestring = MultiLineString(self.lines)
        else:
            self.linestring = LineString(self.coords[:, :2])
        self.Area = self.linestring.buffer(self.length/2).area
        self.Bounds = self.linestring.bounds

    def results(self):
        chars = set(list(self.constants + self.variables))
        avgs = dict()

        for i in chars:
            avgs[i] = re.findall(
                '(?<=' + re.escape(i) + ').*?(?=' + re.escape(i) + ')', self.l_string)

        for i in chars:
            if len(avgs[i]) == 0:
                avgs[i] = 0.0
            else:
                avgs[i] = sum([len(i) for i in avgs[i]])/len(avgs[i])
        try:
            maxF = max(len(s) for s in re.findall(r'[F]+', self.l_string))
        except:
            maxF = 0

        try:
            maxP = max(len(s) for s in re.findall(r'[+]+', self.l_string))
        except:
            maxP = 0

        try:
            maxM = max(len(s) for s in re.findall(r'[-]+', self.l_string))
        except:
            maxM = 0

        self.perF = self.l_string.count('F')/len(self.l_string)
        self.F = self.l_string.count('F')
        self.perP = self.l_string.count('+')/len(self.l_string)
        self.perM = self.l_string.count('-')/len(self.l_string)
        self.perX = self.l_string.count('X')/len(self.l_string)
        if '[' in self.constants:
            self.perB = self.l_string.count('[')/len(self.l_string)
            self.perN = self.l_string.count(']')/len(self.l_string)
        self.maxF = maxF
        self.maxP = maxP
        self.maxM = maxM
        self.avgF = avgs.get('F')
        self.avgP = avgs.get('+')
        self.avgM = avgs.get('-')

        self.comp = np.linalg.norm(self.bounds)

        self.fitness = self.area/self.comp

        self.ratio = np.array([
            (self.choices.count(1)/len(self.choices)) * self.fitness,
            (self.choices.count(2)/len(self.choices)) * self.fitness,
        ])

        self.rules = list(self.rules['X'].values())
