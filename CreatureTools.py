import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from descartes.patch import PolygonPatch
from scipy.spatial.transform import Rotation
from math import radians, cos, sin
import re
from collections import defaultdict
from itertools import islice
from statistics import mean

"""
Looking at what we can do with L-systems, can I create a base class?

https://en.wikipedia.org/wiki/L-system#Example_1:_Algae

http://paulbourke.net/fractals/lsys/

Symbols The following characters have a geometric interpretation.

Character        Meaning
   F	         Move forward by line length drawing a line
   f	         Move forward by line length without drawing a line
   +	         Turn left by turning angle
   -	         Turn right by turning angle
   |	         Reverse direction (ie: turn by 180 degrees)
   [	         Push current drawing state onto stack
   ]	         Pop current drawing state from the stack
   #	         Increment the line width by line width increment
   !	         Decrement the line width by line width increment
   @	         Draw a dot with line width radius
   {	         Open a polygon
   }	         Close a polygon and fill it with fill colour
   >	         Multiply the line length by the line length scale factor
   <	         Divide the line length by the line length scale factor
   &	         Swap the meaning of + and -
   (	         Decrement turning angle by turning angle increment
   )	         Increment turning angle by turning angle increment


"""


class LSystem:
    """
    A builder for an L-system. Generates and returns l_grams for a given set
    of rules.
    Methods
    --------

    """

    def __init__(self,
                 variables,
                 constants,
                 axioms,
                 rules,
                 num_iterations):
        """
        Initialises a simple L-system
        Parameters
        ----------
        num_iterations : int
            number of iterations on the L-sytem
        variables : str
            a string containing all of the letters that take part in the
            recursion. These letters should also have associated rules.
        constants : str or None
            a string containing all the letters that do not take part in the
            recursion. These letters will not have an associated rule
        axioms : str
            The initial character string
        rules : dict
            a dictionary containing the rules for recursion. This is a
            dictionary of listing the letter replacement in the recursion.
            eg.
            {"A": "AB",
            "B": "A"}
        """
        self.num_iterations = num_iterations
        self.rules = rules
        self.axioms = axioms
        self.constants = constants
        self.variables = variables

    def _update_l_string(self, l_string):
        """
        Apply the recursive L-system rules to the current gram
        Parameters
        ----------
        l_string : str
            the imput
        Returns
        -------
        l_string : str
            an updated L_string
        """
        if len(l_string) is not 0:
            l_string = "".join([self.rules.get(c, c) for c in l_string])
        else:
            l_string = l_string + self.axioms

        return l_string

    def generate_new(self):
        """
        Iterate through the recursive L-system num_iterations times and
        return an updated gram.
        Parameters
        ----------

        Returns
        -------
        l_string : str
            L_string gram iterated num_iterations times

        """
        l_string = self.axioms
        for _ in range(self.num_iterations):
            l_string = self._update_l_string(l_string)

        return l_string


class LSystemStochastic(LSystem):
    """
    a base class for an L-system. Contains methods for single and multiple
    recursions. With probabilities.
    """

    def __init__(self,
                 variables,
                 constants,
                 axioms,
                 rules,
                 num_iterations):
        """
        Initialises a simple L-system
        Parameters
        ----------
        num_iterations : int
            the number of iterations on the L-system
        variables : str
            a string containing all of the letters that take part in the
            recursion. These letters should also have associated rules.
        constants : str or None
            a string containing all the letters that do not take part in the
            recursion. These letters will not have an associated rule
        axioms : str
            The initial character string
        rules : dict
            a dictionary containing the rules for recursion. This is a
            dictionary of list with letter replacement in the recursion.
            eg.
            {"A": ["AB", "BA"],
            "B": ["A", "B"]}
        """
        LSystem.__init__(self,
                         variables=variables,
                         constants=constants,
                         axioms=axioms,
                         rules=rules,
                         num_iterations=num_iterations)

    def _update_l_string(self, l_string):
        """
        Apply the recursive L-system rules to the current gram
        Returns
        -------
        None

        """
        if len(l_string) is not 0:
            l_string = ''.join([self.next_char(c) for c in l_string])
        else:
            l_string = l_string + self.axioms

        return l_string

    def next_char(self, c):
        rule = self.rules.get(c, c)
        if len(rule) > 1:
            return np.random.choice(self.rules.get(c, ["Other"])["options"],
                                    p=self.rules.get(c, ["Other"])[
                                        "probabilities"])
        else:
            return rule


class BuilderBase:
    """
    WOrk needs to be done o convert the L-strings into a set of coordinates
    or coords. This class is a helper class and presumes that it will be
    inherited by a class that also inherites from the L-System class.
    """

    def __init__(self,
                 # lstring,
                 point,
                 vector,
                 length,
                 angle,
                 len_scale_factor=1,
                 angle_inc=0):
        """

        Parameters
        ----------
        angle_inc : float or int
            the angle in degrees for changes in direction.
        len_scale_factor : float
            the scale factor for shrinking and growing line length
        point : array like
            the starting point for the l-system
        vector : array like
            the initial direction for the system
        length : float
            the length of a segemnt
        angle : float
            the angle of deviation
        """
        # self.lstring = lstring
        self.angle_base = angle
        self.point_base = point
        self.vector_base = vector
        self.length_base = length
        self.length_scale_factor_base = len_scale_factor
        self.turning_angle_inc_base = angle_inc
        # self.buffer_radius_base = buffer_radius
        self.mapping = {"$": self.start_of_string,
                        "~": self.end_of_string,
                        "F": self.move_forward_draw,
                        "f": self.move_forward_no_draw,
                        "+": self.rotate_left,
                        "-": self.rotate_right,
                        "|": self.reverse_direction,
                        ">": self.multiply_line_length,
                        "<": self.divide_line_length,
                        "&": self.reverse_rotation,
                        "(": self.decriment_angle,
                        "1": self.move_forward_draw,
                        "0": self.move_forward_draw,
                        "[": self.push_to_stack,
                        "]": self.pop_from_stack}
        self.angle = None
        self.point = None
        self.vector = None
        self.length = None
        self.length_scale_factor = None
        self.turning_angle_inc = None
        # self.buffer_radius = None
        self.point_list = None
        self.active_chars = None
        self.control_chars = None
        self.stack = None
        self.coords = None

    def reset_params(self):
        self.angle = self.angle_base
        self.point = self.point_base
        self.vector = self.vector_base
        self.length = self.length_base
        self.length_scale_factor = self.length_scale_factor_base
        self.turning_angle_inc = self.turning_angle_inc_base
        # self.buffer_radius = self.buffer_radius_base
        self.point_list = []
        self.active_chars = None
        self.control_chars = None
        self.stack = []
        self.coords = []

    def get_active_sequence(self, lstring):
        """
        takes the l-string provided and strips out the characters that are
        not associated with creating or actioning the parts. Essentially
        taking the dna and defining the phenotype.
        Returns
        -------

        """
        self.control_chars = ''.join(self.mapping.keys())
        self.active_chars = ''.join([x for x in lstring if x in
                                     self.control_chars])
        self.active_chars = "$" + self.active_chars + "~"

    def build_point_list(self, lstring):
        """
        reads the l-string active componets and finds all of the coordinates.
        Returns
        -------

        """
        self.reset_params()
        self.get_active_sequence(lstring)
        for letter in self.active_chars:
            self.mapping[letter]()

        return self.coords

    def move_forward_draw(self):
        """
        moves forward along current vector by the specified distance. The
        point is appended to the current points list.
        Returns
        -------

        """
        self.vector = self.length * (self.vector / np.linalg.norm(self.vector))
        self.point = self.point + self.vector
        self.point_list.append(self.point)

    def move_forward_no_draw(self):
        """
        Moves forward along the current vector by the specified distance. No
        point is added to the current list The current point list is closed
        and a new one is started.
        Returns
        -------

        """
        if len(self.point_list) > 1:
            self.coords.append(self.point_list)
        self.vector = self.length * (self.vector / np.linalg.norm(self.vector))
        self.point = self.point + self.vector
        self.point_list = [self.point]

    def rotate_left(self):
        """
        rotates the current vector counter clockwise and updates the current
        vector
        Returns
        -------

        """
        r = Rotation.from_euler('z', self.angle, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def rotate_right(self):
        """
        rotates the current vector clockwise and updates the current vector
        Returns
        -------

        """
        r = Rotation.from_euler('z', -self.angle, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def reverse_direction(self):
        """
        reverses and updates the current vector. Similar to a rotation of
        180 degrees

        Returns
        -------

        """
        r = Rotation.from_euler('z', -self.angle, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def multiply_line_length(self):
        """
        multiplies the current line length by the scale factor
        Returns
        -------

        """
        self.length = self.length_scale_factor * self.length

    def divide_line_length(self):
        """
        devides the current line length by the scale factor
        Returns
        -------

        """
        self.length = self.length / self.length_scale_factor

    def reverse_rotation(self):
        temp = self.mapping["+"]
        self.mapping["+"] = self.mapping["-"]
        self.mapping["-"] = temp

    def decriment_angle(self):
        self.angle -= self.turning_angle_inc

    def increment_angle(self):
        self.angle += self.turning_angle_inc

    def push_to_stack(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.stack.append([self.point,
                           self.vector,
                           self.length,
                           self.turning_angle_inc,
                           self.length_scale_factor])

    def pop_from_stack(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """

        if len(self.point_list) > 1:
            self.coords.append(self.point_list)
        self.point, self.vector, self.length, self.turning_angle_inc,  \
<<<<<<< HEAD
            self.length_scale_factor = self.buffer.pop(-1)
||||||| merged common ancestors
        self.length_scale_factor = self.buffer.pop(-1)
=======
        self.length_scale_factor = self.stack.pop(-1)
>>>>>>> upstream/master
        self.point_list = [self.point]

    def end_of_string(self):
        """
        I have a way to build up multiple coords, but the last on gets lost.
        so I use this to append the last segment to the coords list.
        Returns
        -------

        """
        if len(self.point_list) > 1:
            self.coords.append(self.point_list)

    def start_of_string(self):
        """
        initialises the point list
        Returns
        -------

        """
        self.point_list = [self.point]


class BuilderFR(BuilderBase):
    """
    Augments the builder base and replaces some of the methods.
    """

    def __init__(self, lstring,  point, vector, length, angle):
        """

        Parameters
        ----------
        point : array like
            the starting point for the l-system
        vector : array like
            the initial direction for the system
        length : float
            the length of a segemnt
        angle : float
            the angle of deviation
        """
        BuilderBase.__init__(self,
                             lstring,
                             point,
                             vector,
                             length,
                             angle)

    def push_to_stack(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.stack.append([self.point, self.vector])
        self.rotate_left()

    def pop_from_stack(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """

        if len(self.point_list) > 1:
            self.coords.append(self.point_list)
        self.point, self.vector = self.stack.pop(-1)
        self.point_list = [self.point]
        self.rotate_right()


class Plotter:
    """
    Adds some plotting tools for networks. This class is a helper class and
    presumes that it will be inherited by a class that also inherites from the
     L-System class.
    """
<<<<<<< HEAD

    def __init__(self):
        pass
||||||| merged common ancestors
    def __init__(self):
        pass
=======
    def __init__(self, feed_radius):
        self.feed_radius = feed_radius
>>>>>>> upstream/master

    def simple_plot(self, coords):
        """
        A plotting tool for a single line
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = LineString(coords)

<<<<<<< HEAD
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(
            dilated, facecolor='#99ccff', edgecolor='#6699cc')
||||||| merged common ancestors
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
=======
        dilated = line.buffer(self.feed_radius)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
>>>>>>> upstream/master
        ax.add_patch(patch1)
        x, y = line.xy
        plt.axis('equal')
        ax.plot(x, y, color='#999999')
        plt.show()

    def multi_line_plot(self, coords):
        """
        a plotting tool for branching creatures
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = MultiLineString(coords)

<<<<<<< HEAD
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(
            dilated, facecolor='#99ccff', edgecolor='#6699cc')
||||||| merged common ancestors
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
=======
        dilated = line.buffer(self.feed_radius)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
>>>>>>> upstream/master
        ax.add_patch(patch1)
        for i in range(len(coords)):
            x, y = line[i].xy
            plt.axis('equal')
            ax.plot(x, y, color='#999999')
            plt.show()

    def plot_with_feed_zones(self):
        """
        a plotting tool for branching creatures
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = MultiLineString(self.coords)

        patches = [PolygonPatch(circ) for circ in self.feed_zones]
        for patch in patches:
            ax.add_patch(patch)
<<<<<<< HEAD
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(
            dilated, facecolor='#99ccff', edgecolor='#6699cc')
||||||| merged common ancestors
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
=======
        dilated = line.buffer(self.feed_radius)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
>>>>>>> upstream/master
        ax.add_patch(patch1)

        for i in range(len(self.coords)):
            x, y = line[i].xy
            plt.axis('equal')
            ax.plot(x, y, color='#999999')
            plt.show()


class Environment:
    """
    A class that contextualises the attributes of the creature in the
    environment.
    some of the things I am interested in are the length and area of the
    creature so that I can calculate their efficiency
    """
<<<<<<< HEAD

    def __init__(self):
||||||| merged common ancestors
    def __init__(self):
=======
    def __init__(self, feed_radius=0.5):
>>>>>>> upstream/master
        """
        initialise the environment
        """
        self.feed_radius = feed_radius
        # self.creature = None
        # self.creature_feed_zone = None
        # self.creature_length = None
        # self.creature_feed_zone = None
        # self.creature_fitness = None
        # self.feed_zones = []

    # def place_feed_zones(self, feed_zones):
    #     """
    #     place some objectives in the environment. For a start we will simple
    #     place selected regions of rewards rather than simply using the size
    #     to determine fitness.
    #     Parameters
    #     ----------
    #     feed_zones : list of tuples
    #         a list of thrupples containing the x, y positions for the food
    #         source. the final value in the thrupple is the radius of the feed
    #         zone
    #
    #     Returns
    #     -------
    #
    #     """
    #     self.feed_zones = [Point(zone[0], zone[1]).buffer(zone[2]) for zone in
    #                        feed_zones]

    def get_fitness(self, creature_feed_area):
        """

        Returns
        -------

        """
        return creature_feed_area.area

<<<<<<< HEAD
    def get_fitness(self):
        """

        Returns
        -------

        """
        self.creature_fitness = self.creature_feed_zone.intersection(
            self.feed_zones)
||||||| merged common ancestors
    def get_fitness(self):
        """

        Returns
        -------

        """
        self.creature_fitness = self.creature_feed_zone.intersection(self.feed_zones)
=======
    def expose_to_environment(self, coords):
        creature = MultiLineString(coords)
        creature_length = creature.length
        creature_feed_area = creature.buffer(self.feed_radius)
        fitness = self.get_fitness(creature_feed_area)
>>>>>>> upstream/master

        return creature_length, creature_feed_area, fitness


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
