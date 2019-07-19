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
        self.rules = params.get('rules')
        self.l_string = params.get('axiom')
        self.constants = params.get('constants')
        self.variables = params.get('variables')
        self.angle = radians(params.get('angle'))
        self.recur(params.get('num_char'))
        self.length = params.get('length')
        self.mapper()
        self.layout()
        self.results()

    def recur(self, n):
        for _ in range(n):
            self.l_string = ''.join([self.next_char(c) for c in self.l_string])
        self.l_string = self.l_string.replace('X', '')

    def next_char(self, c):
        rule = self.rules.get(c, c)
        if len(rule) > 1:
            return np.random.choice(self.rules.get(c, ["Other"])["options"],
                                    p=self.rules.get(c, ["Other"])[
                                        "probabilities"])
        else:
            return rule

    def mapper(self):
        """Converts L-string to coordinates

        Returns
        -------
        List
            List of coordinates
        """

        num_chars = len(self.l_string)

        coords = np.zeros((num_chars + 1, 3), np.double)

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
                coords[i] = (coords[i-1] + (self.length * curr_vec))
                i += 1

            if c == '-':
                curr_vec = np.dot(curr_vec, (-1*rotVec))

            if c == '+':
                curr_vec = np.dot(curr_vec, rotVec)

        coords = np.delete(coords, np.s_[i:], 0)
        self.coords = coords

    def layout(self):
        self.linestring = LineString(self.coords[:, :2])
        self.area = self.linestring.buffer(self.length/2).area
        self.bounds = self.linestring.bounds

    def results(self):
        chars = set(list(self.l_string))
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
            maxF = max(len(s) for s in re.findall(r'F+', self.l_string))
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
        self.perP = self.l_string.count('+')/len(self.l_string)
        self.perM = self.l_string.count('-')/len(self.l_string)
        self.maxF = maxF
        self.maxP = maxP
        self.maxM = maxM
        self.avgF = avgs.get('F')
        self.avgP = avgs.get('+')
        self.avgM = avgs.get('-')


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
        self.area = self.linestring.buffer(self.length/2).area
        self.bounds = self.linestring.bounds

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
