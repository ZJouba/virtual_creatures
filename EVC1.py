import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from descartes.patch import PolygonPatch
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def next_point(angle, vec, dist, point):
    """
    Takes the current point and vector and finds the next point and vector
    Parameters
    ----------
    angle : float
        the angle of rotation
    vec : array like
        the current 2d vector
    dist : float
        the projection distance for the next point
    point : array like
        the 2d starting point for the projection

    Returns
    -------
    point : array
        the projected point in 2d
    vec : array
        the updated vector

    """
    r = Rotation.from_euler('z', angle, degrees=True)
    vec = np.append(vec, [0])
    vec = r.apply(vec)[:2]
    vec = dist * (vec / np.linalg.norm(vec))
    point = point + vec
    return point, vec


def draw_phenotype(phenotype):
    """
    Sketch out the worm.
    Returns
    -------

    """
    fig = plt.figure(1, figsize=(5, 5), dpi=180)

    line = LineString(phenotype)
    ax = fig.add_subplot(111)
    dilated = line.buffer(0.5)
    patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
    ax.add_patch(patch1)
    x, y = line.xy
    plt.axis('equal')
    ax.plot(x, y, color='#999999')

    plt.show()


class Vermiculus(object):
    """
    Makes a worm like creature
    """
    def __init__(self,
                 instance_number,
                 starting_point=np.array([0, 0]),
                 starting_vector=np.array([0, 1]),
                 dna_length=60,
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
        self.starting_point = starting_point
        self.starting_vector = starting_vector
        self.dna_length = dna_length
        self.unit_length = unit_length
        self.unit_width = unit_width
        # self.proteins = {"F": 0.0,
        #                  "L": -1 * np.pi / 4,
        #                  "R": np.pi / 4}
        self.proteins = {"F": 0.0,
                         "L": 45,
                         "R": -45}
        self.amino_acids = list(self.proteins.keys())
        self.amino_probabilities = [0.4, 0.3, 0.3]

        self.phenotype = [self.starting_point]
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

    def _translate_dna(self):
        """
        read the DNA sequence and build up a string of segment end points.
        Returns
        -------

        """
        cv = self.starting_vector
        cp = self.starting_point
        points = [cp]
        for acid in self.dna:
            cp, cv = next_point(self.proteins[acid],
                                cv,
                                self.unit_length,
                                cp)
            points.append(cp)

        self.phenotype = points

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


class Radix(Vermiculus):
    """
    Makes a root like structure
    """
    def __init__(self):
        self.x = 1


def scan_population(num):
    areas = []
    for i in tqdm(range(num)):
        worm = Vermiculus(i)
        worm.build_topology()
        areas.append(worm.area)
    plt.hist(areas, bins=50)


def make_one_worm():
    worm = Vermiculus(1)
    worm.build_topology()
    draw_phenotype(worm.phenotype)


if __name__ == "__main__":
    # scan_population(5000)
    make_one_worm()