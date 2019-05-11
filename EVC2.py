import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point
from descartes.patch import PolygonPatch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

"""
Looking at what we can do with L-systems, can I create a base class?
"""


class L_System:
    def __init__(self,
                 variables,
                 constants,
                 axioms,
                 rules):
        """
        Initialises a simple L-system
        Parameters
        ----------
        variables
        constants
        axioms
        rules
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


class BuilderBase:
    def __init__(self, point, vector, length):
        # self.l_string = l_string
        self.point = point
        self.vector = vector
        self.length = length
        self.point_list = [point]
        self.mapping = {"F": self.move_forward,
                        "+": self.rotate_right,
                        "-": self.rotate_left}
        self.active_chars = None
        self.control_chars = None

    def move_forward(self):
        self.vector = self.length * (self.vector / np.linalg.norm(self.vector))
        self.point = self.point + self.vector
        self.point_list.append(self.point)

    def rotate_left(self):
        r = Rotation.from_euler('z', 90, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def rotate_right(self):
        r = Rotation.from_euler('z', -90, degrees=True)
        vec = np.append(self.vector, [0])
        self.vector = r.apply(vec)[:2]

    def get_active_sequence(self):
        self.control_chars = ''.join(self.mapping.keys())
        self.active_chars = ''.join([x for x in self.l_string if x in
                                     self.control_chars])

    def build_point_list(self):
        self.get_active_sequence()
        for letter in self.active_chars:
            self.mapping[letter]()
        # return self.point_list


class Plotter:
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


class BinaryTree(L_System):
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
                         1.0)
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
                         1.0)
        Plotter.__init__(self)



class FractalPlant(L_System):
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


if __name__ == "__main__":
    # sys = DragonCurve()
    # sys = KochCurve()
    sys = BinaryTree()
    sys.recur_n(3)
    # sys.build_point_list()
    # sys.simple_plot()

