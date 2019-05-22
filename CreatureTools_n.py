import numpy as np
from shapely.geometry import LineString
from math import radians, cos, sin, pi
import re


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
        self.angle = params.get('angle')
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
        theta = 1.570

        if isinstance(self.angle, int):
            randomAngle = False
        elif self.angle == 'random':
            randomAngle = True

        def getAngle():
            if randomAngle:
                return (np.random.uniform(0, 0.5) * pi)
            else:
                return radians(self.angle)

        num_chars = len(self.l_string)

        coords = np.zeros((num_chars + 1, 3), np.double)

        def makeRotMat(theta):
            rotMat = np.array((
                (cos(theta), -sin(theta), 0),
                (sin(theta), cos(theta), 0),
                (0, 0, 1)
            ))
            return rotMat

        rotVec = makeRotMat(theta)

        begin_vec = np.array((1, 0, 0), np.float64)
        i = 1

        for c in self.l_string:
            if c == 'F':
                next_vec = np.dot(rotVec, begin_vec)
                coords[i] = (
                    coords[i-1] + (self.length * next_vec)
                )
                i += 1
                begin_vec = next_vec

            if c == '-':
                theta = theta - getAngle()
                rotVec = makeRotMat(theta)

            if c == '+':
                theta = theta + getAngle()
                rotVec = makeRotMat(theta)

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
