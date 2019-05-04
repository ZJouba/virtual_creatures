import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from descartes.patch import PolygonPatch
from tqdm import tqdm
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types
from numba import jit, njit


# spec = [
#     ('instance_number', int32),               # a simple scalar field
#     ('area', float32[:]),          # an array field
#     ('dna_length', int32),               # a simple scalar field
#     ('angle', float32),          # an array field
#     ('unit_length', float32),          # an array field
#     ('unit_width', float32),          # an array field
# ]
#
# @jitclass(spec)
class Vermiculus(object):
    """
    Makes a worm like creature
    """
    amino_acids = ["F", "L", "R"]
    amino_probabilities = [0.4, 0.3, 0.3]
    proteins = {"F": 0.0,
                "L": -1 * np.pi / 4,
                "R": np.pi / 4}

    def __init__(self,
                 instance_number,
                 starting_point=np.array([0, 0]),
                 starting_vector=np.array([0, 1]),
                 dna_length=60,
                 # angle=np.pi / 4,
                 unit_length=1.0,
                 unit_width=1.0,
                 ):
        """
        Initialise a worm
        Parameters
        ----------
        starting_point : array like
            a 2d vector containing the strarting point for the worm
        instance_number: int
            the index for the instance of the worm
        """
        self.instance_number = instance_number
        self.current_point = starting_point
        self.current_vector = starting_vector
        self.dna_length = dna_length
        # self.angle = angle
        self.unit_length = unit_length
        self.unit_width = unit_width
        self.phenotype = [self.current_point]
        self.dna = self.transcribe_dna()
        self.topology = None
        self.topology_dilated = None
        self.area = None

    def transcribe_dna(self):
        """
        Build a DNA string for the worm that is N elements long by selecting
        from the available amino acids
        Returns
        -------
            A string N elements long
        """
        return ''.join(np.random.choice(self.amino_acids,
                                        p=self.amino_probabilities
                                        ) for _ in range(self.dna_length))

    def update_vector(self, amino_acid):
        """
        Reading the next amino acid in the DNA string sequence. Depending on
        the behavior indicated by the protein the next point in the sequence.
        Parameters
        ----------
        amino_acid: char
            a single char indicating the next amino acid in the sequence.
        Returns
        -------

        """
        vec = self.current_vector
        vec = self.rotate_vector(vec, self.proteins[amino_acid])
        self.current_vector = vec / np.linalg.norm(vec)

    def update_point(self, amino_acid):
        """
        Read the next amino acid in the DNA sequence and create a new worm
        segment.
        Parameters
        ----------
        amino_acid: char
            a single char indicating the next amino acid in the sequence.
        Returns
        -------

        """
        point = self.current_point
        self.update_vector(amino_acid)
        vec = self.current_vector
        self.current_point = point + vec

    def _translate_dna(self):
        """
        read the DNA sequence and build up a string of segment end points.
        Returns
        -------

        """
        for acid in self.dna:
            self.update_point(acid)
            self.phenotype.append(self.current_point)

    def build_topology(self):
        """
        Translate the phenotype to the topology
        Returns
        -------

        """
        self._translate_dna()
        self.topology = LineString(self.phenotype)
        self.topology_dilated = self.topology.buffer(self.unit_width)
        self.area = self.topology_dilated.area

    def draw_phenotype(self):
        """
        Sketch out the worm.
        Returns
        -------

        """
        fig = plt.figure(1, figsize=(10, 4), dpi=180)

        line = LineString(self.phenotype)
        ax = fig.add_subplot(121)
        dilated = line.buffer(0.5)
        patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
        ax.add_patch(patch1)
        x, y = line.xy
        ax.plot(x, y, color='#999999')

        plt.show()

    @staticmethod
    def rotate_vector(xy, radians):
        """
        Use numpy to build a rotation matrix and take the dot product.
        Parameters
        ----------
        xy: array like
            an array or tuple containing the coordinates of one point in the
            worm
        radians: float
            the radian angle to rotate the vector.
        Returns
        -------
            an array containing the rotated and normalized vector.
        """
        x, y = xy
        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])
        return np.array([float(m.T[0]), float(m.T[1])])

@jit
def get_areas(population):
    area = []
    for i in range(population):
        worm = Vermiculus(i)
        worm.build_topology()
        area.append(worm.area)
    return area


if __name__ == "__main__":
    # pop = 100
    # areas = np.zeros(pop, dtype=np.float32)
    # areas = get_areas(pop)
    areas = []
    for i in tqdm(range(50000)):
        worm = Vermiculus(i)
        worm.build_topology()
        areas.append(worm.area)
    # worm.build_topology()
    # worm = Vermiculus(1)
    # # print(worm.dna)
    # worm.build_topology()
    # # print(worm.phenotype)
    # worm.draw_phenotype()
    # # print(worm.area)
    # # print(worm.topology.is_simple)
    plt.hist(areas, bins=50)