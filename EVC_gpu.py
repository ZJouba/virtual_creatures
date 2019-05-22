"""
an attempt to run a creature population using GPU
"""

class LSystem:
    """
    a base class for an L-system. Contains methods for single and multiple
    recursions.

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
