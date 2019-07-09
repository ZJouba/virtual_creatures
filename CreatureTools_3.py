import numpy as np
from shapely.geometry import LineString
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
        self.choices = []
        self.chars = params.get('chars')
        self.length = params.get('length')
        self.angle = radians(params.get('angle'))
        self.prune = params.get('prune')

        if not 'L-string' in params:
            self.rules = params.get('rules')
            self.l_string = params.get('axiom')
            self.constants = params.get('constants')
            self.recur(params.get('recurs'))

        else:
            self.l_string = params.get('L-string')

        self.mapper()
        self.layout()
        self.results()

    def recur(self, iters):
        for _ in range(iters):
            if (self.l_string.count('X') == 0) or (len(self.l_string) > (self.chars*2)):
                break
            else:
                self.l_string = ''.join([self.next_char(c) for c in self.l_string])
        
        if self.prune:
            if len(self.l_string) > self.chars:
                self.l_string = self.l_string[:500]

        # self.l_string = self.l_string.replace('X', '')
        
    def next_char(self, c):
        rule = self.rules.get(c, c)
        if not rule == c:
            key, choice = random.choice(list(self.rules.get(c).items()))
            self.choices.append(key)
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
        self.area = self.linestring.buffer(0.499999).area
        self.bounds = self.linestring.bounds

    def results(self):

        self.ratio = np.array([
            (self.choices.count(1)/len(self.choices)) * self.area,
            (self.choices.count(2)/len(self.choices)) * self.area,
        ])

        self.rules = list(self.rules['X'].values())
