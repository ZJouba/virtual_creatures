# Creatures.py
"""
a collection of creatures to explore
"""
from CreatureTools import *


class Alga(LSystem):
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
    def __init__(self, n):
        LSystem.__init__(self,
                         "AB",
                         None,
                         "A",
                         {"A": "AB",
                          "B": "A"})
        self.recur_n(n)


class BinaryTree(LSystem, BuilderFR, Plotter):
    """
    Generate a binary tree L-system
    Tests
    -------
    axiom: 0
    1st recursion: 	1[0]0
    2nd recursion: 	11[1[0]0]1[0]0
    3rd recursion: 	1111[11[1[0]0]1[0]0]11[1[0]0]1[0]0
    """
    def __init__(self, n):
        LSystem.__init__(self,
                         "01",
                         "[]",
                         "0",
                         {"1": "11",
                          "0": "1[0]0"})
        self.recur_n(n)
        BuilderFR.__init__(self,
                           self.l_string,
                           np.array([0, 0]),
                           np.array([0, 1]),
                           1.0,
                           45)
        Plotter.__init__(self)


class CantorSet(LSystem):
    """
    Generate a Cantor Set L-system
    Tests
    -------
    n = 0 : A
    n = 1 : ABA
    n = 2 : ABABBBABA
    """
    def __init__(self, n):
        LSystem.__init__(self,
                         "AB",
                         None,
                         "A",
                         {"A": "ABA",
                          "B": "BBB"})
        self.recur_n(n)


class KochCurve(LSystem, BuilderBase, Plotter):
    """
    Generate a Koch Curve L-system
    Tests
    -------
    n = 0 : F
    n = 1 : F+F−F−F+F
    n = 2 : F+F−F−F+F+F+F−F−F+F−F+F−F−F+F−F+F−F−F+F+F+F−F−F+F
    """
    def __init__(self, n):
        LSystem.__init__(self,
                         "F",
                         "+-",
                         "F",
                         {"F": "F+F-F-F+F", })
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([1, 0]),
                             1.0,
                             90)
        Plotter.__init__(self)


class SierpinskiTriangle(LSystem):
    """
    Generate a Sierpinski Triangle L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         "FG",
                         "+-",
                         "F-G-G",
                         {"F": "F−G+F+G−F",
                          "G": "GG"})
        self.recur_n(n)


class DragonCurve(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         "XY",
                         "F+-",
                         "FX",
                         {"X": "X+YF+",
                          "Y": "-FX-Y"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             90)
        Plotter.__init__(self)


class FractalPlant(LSystem, BuilderBase, Plotter):
    """
    Generate a Fractal Plant L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self, "XF",
                         "+-[]",
                         "X",
                         {"X": "F+[[X]-X]-F[-FX]+X",
                          "F": "FF"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             15)
        Plotter.__init__(self)


class Worm(LSystemStochastic, BuilderBase, Plotter):
    """
    Generate a Worm L-system
    Tests
    -------
    ,
                 ,

    """
    def __init__(self, n):
        LSystemStochastic.__init__(self,
                                   variables="X",
                                   constants="F+-",
                                   axioms="FX",
                                   rules={"X": {"options": ["+FX", "-FX"],
                                                "probabilities": [0.5, 0.5]}})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             25)
        Plotter.__init__(self)


class ZeroL(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         variables="F",
                         constants="",
                         axioms="F+F+F+F",
                         rules={"F": "F+F-F-FF+F+F-F"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([1, 0]),
                             1.0,
                             90)
        Plotter.__init__(self)


class Bricks(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         variables="F",
                         constants="",
                         axioms="F+F+F+F",
                         rules={"F": "FF+F-F+F+FF"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([1, 0]),
                             1.0,
                             90)
        Plotter.__init__(self)


class Bush(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         variables="XY",
                         constants="F+-[]",
                         axioms="Y",
                         rules={"X": "X[-FFF][+FFF]FX",
                                "Y": "YFX[+Y][-Y]"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             25.7)
        Plotter.__init__(self)


class Bush2(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """

    def __init__(self, n):
        LSystem.__init__(self,
                         variables="F",
                         constants="F+-[]",
                         axioms="F",
                         rules={"F": "FF+[+F-F-F]-[-F+F+F]", })
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             25.7)
        Plotter.__init__(self)


class Bush3(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """

    def __init__(self, n):
        LSystem.__init__(self,
                         variables="VWXYZ",
                         constants="F+-[]",
                         axioms="VZFFF",
                         rules={"V": "[+++W][---W]YV",
                                "W": "+X[-W]Z",
                                "X": "-W[+X]Z",
                                "Y": "YZ",
                                "Z": "[-FFF][+FFF]F"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             20)
        Plotter.__init__(self)


class Bush4(LSystem, BuilderBase, Plotter):
    """
    Generate a Dragon Curve L-system
    Tests
    -------

    """

    def __init__(self, n):
        LSystem.__init__(self,
                         variables="X",
                         constants="F+-[]",
                         axioms="FX",
                         rules={"X": ">[-FX]+FX"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             40,
                             lenght_scale_factor=0.5)
        Plotter.__init__(self)


class Leaf(LSystem, BuilderBase, Plotter):
    """
    Generate a leaf L-system
    Tests
    -------

    """
    def __init__(self, n):
        LSystem.__init__(self,
                         variables="FabXY",
                         constants="<>+-[]",
                         axioms="a",
                         rules={"F": ">F<",
                                "a": "F[+X]Fb",
                                "b": "F[-Y]Fa",
                                "X": "a",
                                "Y": "b"})
        self.recur_n(n)
        BuilderBase.__init__(self,
                             self.l_string,
                             np.array([0, 0]),
                             np.array([0, 1]),
                             1.0,
                             45,
                             lenght_scale_factor=1.3)
        Plotter.__init__(self)


if __name__ == "__main__":
    # sys = DragonCurve(10)
    # sys = KochCurve(3)
    # sys = BinaryTree(4)
    sys = Worm(100)
    # sys = FractalPlant(4)
    # sys = ZeroL(2)
    # sys = Bricks(3)
    # sys = Bush(5)
    # sys = Bush2(4)
    # sys = Bush3(8)
    # sys = Bush4(5)
    # sys = Leaf(9)

    sys.build_point_list()
    # sys.simple_plot()
    sys.multi_line_plot()

