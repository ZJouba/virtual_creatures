import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point
from descartes.patch import PolygonPatch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

"""
Looking at what we can do with L-systems, can I create a base class?

https://en.wikipedia.org/wiki/L-system#Example_1:_Algae
"""


class L_System:
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
            self.l_string = "".join([self.rules.get(c, c) for c in self.l_string])
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


class RandomBuild:
    """
    A string builder similar to an L-system, but without recursive replacement
    """
    def __init__(self,
                 variables,
                 constants,
                 axioms):
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
        """
        self.axioms = axioms
        self.constants = constants
        self.variables = variables
        self.variable_char = [char for char in variables]
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
            self.l_string = self.l_string + np.random.choice(self.variable_char)
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


class BuilderBase:
    """
    WOrk needs to be done o convert the L-strings into a set of coordinates
    or lines. This class is a helper class and presumes that it will be
    inherited by a class that also inherites from the L-System class.
    """
    def __init__(self, point, vector, length, angle):
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
        self.angle = angle
        self.point = point
        self.vector = vector
        self.length = length
        self.point_list = [point]
        self.mapping = {"~": self.end_of_string,
                        "F": self.move_forward,
                        "+": self.rotate_right,
                        "-": self.rotate_left,
                        "1": self.move_forward,
                        "0": self.move_forward,
                        "[": self.push_to_buffer,
                        "]": self.pop_from_buffer}
        self.active_chars = None
        self.control_chars = None
        self.buffer = []
        self.lines = []

    def get_active_sequence(self):
        """
        takes the l-string provided and strips out the characters that are
        not associated with creating or actioning the parts. Essentially
        taking the dna and defining the phenotype.
        Returns
        -------

        """
        self.control_chars = ''.join(self.mapping.keys())
        self.active_chars = ''.join([x for x in self.l_string if x in
                                     self.control_chars])
        self.active_chars = self.active_chars + "~"

    def build_point_list(self):
        """
        reads the l-string active componets and finds all of the coordinates.
        Returns
        -------

        """
        self.get_active_sequence()
        for letter in self.active_chars:
            self.mapping[letter]()

    def move_forward(self):
        """
        creates a new point along the current vector and appends it to the list
        Returns
        -------

        """
        self.vector = self.length * (self.vector / np.linalg.norm(self.vector))
        self.point = self.point + self.vector
        self.point_list.append(self.point)

    def rotate_left(self):
        """
        rotates the current vector counter clockwise
        Returns
        -------

        """
        r = Rotation.from_euler('z', self.angle, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def rotate_right(self):
        """
        rotates the current vector clockwise
        Returns
        -------

        """
        r = Rotation.from_euler('z', -self.angle, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

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
        self.point, self.vector = self.buffer.pop()
        self.lines.append(self.point_list)
        self.point_list = [self.point]
        self.rotate_right()

    def end_of_string(self):
        """
        I have a way to build up multiple lines, but the last on gets lost.
        so I use this to append the last segment to the lines list.
        Returns
        -------

        """
        if len(self.point_list) > 1:
            self.lines.append(self.point_list)


class BuilderPlant(BuilderBase):
    """
    Augments the builder base and replaces some of the methods.
    """
    def __init__(self, point, vector, length, angle):
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
                             point,
                             vector,
                             length,
                             angle)

        self.mapping = {"~": self.end_of_string,
                        "F": self.move_forward,
                        "+": self.rotate_right,
                        "-": self.rotate_left,
                        "1": self.move_forward,
                        "0": self.move_forward,
                        "[": self.push_to_buffer,
                        "]": self.pop_from_buffer}

    def push_to_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.buffer.append([self.point, self.vector])
        # self.rotate_left()

    def pop_from_buffer(self):
        """
        append the current point and vector to a list for later
        Returns
        -------

        """
        self.point, self.vector = self.buffer.pop(-1)
        # if len(self.point_list) > 1:
        #     self.lines.append(self.point_list)
        #     self.point_list = [self.point]
        # self.rotate_right()


class Plotter:
    """
    Adds some plotting tools for networks. This class is a helper class and
    presumes that it will be inherited by a class that also inherites from the
     L-System class.
    """
    def __init__(self):
        pass

    def simple_plot(self):
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = LineString(self.point_list)

        dilated = line.buffer(0.5)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(patch1)
        x, y = line.xy
        plt.axis('equal')
        ax.plot(x, y, color='#999999')
        plt.show()

    def multi_line_plot(self):
        fig = plt.figure(1, figsize=(5, 5), dpi=180)
        ax = fig.add_subplot(111)
        line = MultiLineString(self.lines)

        dilated = line.buffer(0.5)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(patch1)
        for i in range(len(self.lines)):
            x, y = line[i].xy
            plt.axis('equal')
            ax.plot(x, y, color='#999999')
            plt.show()

class Alga(L_System):
    """
    build a algea L-system
    Tests
    -------
    n = 0 : A
    n = 1 : AB
    n = 2 : ABA
    n = 3 : ABAAB
    n = 4 : ABAABABA
    n = 5 : ABAABABAABAAB
    n = 6 : ABAABABAABAABABAABABA
    n = 7 : ABAABABAABAABABAABABAABAABABAABAAB
    """
    def __init__(self):
        L_System.__init__(self, "AB",
                          None,
                          "A",
                          {"A": "AB",
                           "B": "A"})


class BinaryTree(L_System, BuilderBase, Plotter):
    """
    Generate a binary tree L-system
    Tests
    -------
    axiom: 0
    1st recursion: 	1[0]0
    2nd recursion: 	11[1[0]0]1[0]0
    3rd recursion: 	1111[11[1[0]0]1[0]0]11[1[0]0]1[0]0
    """
    def __init__(self):
        L_System.__init__(self, "01",
                          "[]",
                          "0",
                          {"1": "11",
                           "0": "1[0]0"})
        BuilderBase.__init__(self,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             45)
        Plotter.__init__(self)


class CantorSet(L_System):
    """
    Generate a Cantor Set L-system
    Tests
    -------
    n = 0 : A
    n = 1 : ABA
    n = 2 : ABABBBABA
    """
    def __init__(self):
        L_System.__init__(self, "AB",
                          None,
                          "A",
                          {"A": "ABA",
                           "B": "BBB"})


class KochCurve(L_System, BuilderBase, Plotter):
    """
    Generate a Koch Curve L-system
    Tests
    -------
    n = 0 : F
    n = 1 : F+F−F−F+F
    n = 2 : F+F−F−F+F+F+F−F−F+F−F+F−F−F+F−F+F−F−F+F+F+F−F−F+F
    """
    def __init__(self):
        L_System.__init__(self, "F",
                          "+-",
                          "F",
                          {"F": "F+F-F-F+F", })
        BuilderBase.__init__(self,
                             np.array([0, 0]),
                             np.array([1, 0]),
                             1.0,
                             90)
        Plotter.__init__(self)


class SierpinskiTriangle(L_System):
    """
    Generate a Sierpinski Triangle L-system
    Tests
    -------

    """
    def __init__(self):
        L_System.__init__(self, "FG",
                          "+-",
                          "F-G-G",
                          {"F": "F−G+F+G−F",
                           "G": "GG"})


class DragonCurve(L_System, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """
    def __init__(self):
        L_System.__init__(self, "XY",
                          "F+-",
                          "FX",
                          {"X": "X+YF+",
                           "Y": "-FX-Y"})
        BuilderBase.__init__(self,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             90)
        Plotter.__init__(self)



class FractalPlant(L_System, BuilderPlant, Plotter):
    """
    Generate a Fractal Plant L-system
    Tests
    -------

    """
    def __init__(self):
        L_System.__init__(self, "XF",
                          "+-[]",
                          "X",
                          {"X": "F+[[X]-X]-F[-FX]+X",
                           "F": "FF"})
        BuilderPlant.__init__(self,
                              np.array([0, 0]),
                              np.array([0, 1]),
                              1.0,
                              45)
        Plotter.__init__(self)


class Worm(RandomBuild, BuilderBase, Plotter):
    """
    Generate a binary tree L-system
    Tests
    -------
    ,
                 ,

    """
    def __init__(self):
        RandomBuild.__init__(self,
                             variables="F+-",
                             constants="",
                             axioms="F",)
        BuilderBase.__init__(self,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             45)
        Plotter.__init__(self)


class AL(RandomBuild, BuilderBase, Plotter):
    """
    Generate a binary tree L-system
    Tests
    -------
    ,
                 ,

    """
    def __init__(self):
        RandomBuild.__init__(self,
                             variables="F+-",
                             constants="",
                             axioms="F",)
        BuilderBase.__init__(self,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             45)
        Plotter.__init__(self)



if __name__ == "__main__":
    # sys = DragonCurve()
    # sys = KochCurve()
    # sys = BinaryTree()
    # sys = Worm()
    sys = FractalPlant()
    sys.recur_n(2)
    sys.build_point_list()
    # sys.simple_plot()
    sys.multi_line_plot()

