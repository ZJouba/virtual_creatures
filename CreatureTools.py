import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point
from descartes.patch import PolygonPatch
from scipy.spatial.transform import Rotation

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
    a base class for an L-system. Contains methods for single and multiple
    recursions
    """
    def __init__(self,
                 variables,
                 constants,
                 axioms,
                 rules):
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
        axioms : str
            The initial character string
        rules : dict
            a dictionary containing the rules for recursion. This is a
            dictionary of listing the letter replacement in the recursion.
            eg.
            {"A": "AB",
            "B": "A"}
        """
        self.rules = rules
        self.axioms = axioms
        self.constants = constants
        self.variables = variables
        self.l_string = ""

    def _update_product(self):
        """
        internal method for applying the recursive L-System rules. The
        L-System l_string is updated
        Returns
        -------
        None

        """
        if len(self.l_string) is not 0:
            self.l_string = "".join([self.rules.get(c, c)
                                     for c in self.l_string])
        else:
            self.l_string = self.l_string + self.axioms

    def recur_n(self, n):
        """
        iterate through the recursive L-system update n times.
        Parameters
        ----------
        n : int
            number of iterations of the L-System update

        Returns
        -------
        None

        """
        self.l_string = self.axioms
        for _ in range(n):
            self._update_product()


class LSystemStochastic(LSystem):
    """
    a base class for an L-system. Contains methods for single and multiple
    recursions. With probabilities.
    """
    def __init__(self,
                 variables,
                 constants,
                 axioms,
                 rules):
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
                         rules=rules)

    def _update_product(self):
        """
        internal method for applying the recursive L-System rules. The
        L-System l_string is updated
        Returns
        -------
        None

        """
        if len(self.l_string) is not 0:
            self.l_string = ''.join([self.next_char(c) for c in self.l_string])
        else:
            self.l_string = self.l_string + self.axioms

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
                 lstring,
                 point,
                 vector,
                 length,
                 angle,
                 lenght_scale_factor=1,
                 turning_angle_inc=0,
                 buffer_diameter=0.5):
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
        self.buffer_diameter = buffer_diameter
        self.lstring = lstring
        self.angle = angle
        self.point = point
        self.vector = vector
        self.length = length
        self.turning_angle_inc = turning_angle_inc
        self.length_scale_factor = lenght_scale_factor
        self.point_list = []
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
                        "[": self.push_to_buffer,
                        "]": self.pop_from_buffer}
        self.active_chars = None
        self.control_chars = None
        self.buffer = []
        self.coords = []

    def get_active_sequence(self):
        """
        takes the l-string provided and strips out the characters that are
        not associated with creating or actioning the parts. Essentially
        taking the dna and defining the phenotype.
        Returns
        -------

        """
        self.control_chars = ''.join(self.mapping.keys())
        self.active_chars = ''.join([x for x in self.lstring if x in
                                     self.control_chars])
        self.active_chars = "$" + self.active_chars + "~"

    def build_point_list(self):
        """
        reads the l-string active componets and finds all of the coordinates.
        Returns
        -------

        """
        self.get_active_sequence()
        for letter in self.active_chars:
            self.mapping[letter]()

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

    def push_to_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.buffer.append([self.point,
                            self.vector,
                            self.length,
                            self.turning_angle_inc,
                            self.length_scale_factor])

    def pop_from_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """

        if len(self.point_list) > 1:
            self.coords.append(self.point_list)
        self.point, self.vector, self.length, self.turning_angle_inc,  \
        self.length_scale_factor = self.buffer.pop(-1)
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

    def push_to_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.buffer.append([self.point, self.vector])
        self.rotate_left()

    def pop_from_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """

        if len(self.point_list) > 1:
            self.coords.append(self.point_list)
        self.point, self.vector = self.buffer.pop(-1)
        self.point_list = [self.point]
        self.rotate_right()


class Plotter:
    """
    Adds some plotting tools for networks. This class is a helper class and
    presumes that it will be inherited by a class that also inherites from the
     L-System class.
    """
    def __init__(self):
        pass

    def simple_plot(self):
        """
        A plotting tool for a single line
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = LineString(self.coords)

        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(patch1)
        x, y = line.xy
        plt.axis('equal')
        ax.plot(x, y, color='#999999')
        plt.show()

    def multi_line_plot(self):
        """
        a plotting tool for branching creatures
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = MultiLineString(self.coords)

        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(patch1)
        for i in range(len(self.coords)):
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
        dilated = line.buffer(self.buffer_diameter)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
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
    def __init__(self):
        """
        initialise the environment
        """
        self.creature = None
        self.creature_feed_zone = None
        self.creature_length = None
        self.creature_area = None
        self.creature_fitness = None
        self.feed_zones = []

    def place_feed_zones(self, feed_zones):
        """
        place some objectives in the environment. For a start we will simple
        place selected regions of rewards rather than simply using the size
        to determine fitness.
        Parameters
        ----------
        feed_zones : list of tuples
            a list of thrupples containing the x, y positions for the food
            source. the final value in the thrupple is the radius of the feed
            zone

        Returns
        -------

        """
        self.feed_zones = [Point(zone[0], zone[1]).buffer(zone[2]) for zone in
                           feed_zones]

    def get_fitness(self):
        """

        Returns
        -------

        """
        self.creature_fitness = self.creature_feed_zone.intersection(self.feed_zones)

    def expose_to_environment(self):
        self.creature = MultiLineString(self.coords)
        self.creature_length = self.creature.length
        self.creature_feed_zone = self.creature.buffer(self.buffer_diameter)
        self.creature_area = self.creature_feed_zone.area
        self.get_fitness()


