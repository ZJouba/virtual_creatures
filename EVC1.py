import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Vermiculus:
    dna_length = 60
    angle = np.pi/4
    unit_length = 1.0
    amino_acids = ["F", "L", "R"]
    proteins = {"F": 0.0,
                "L": -1 * angle,
                "R": angle}

    def __init__(self, instance_number):
        self.instance_number = instance_number
        self.dna = self.transcribe_dna()
        self.current_point = np.array([0, 0])
        self.current_vector = np.array([1, 0])
        self.phenotype = [self.current_point]

    def transcribe_dna(self):
        return ''.join(np.random.choice(self.amino_acids) for
                       _ in range(self.dna_length))

    def update_vector(self, amino_acid):
        vec = self.current_vector
        vec = self.rotate_vector(vec, self.proteins[amino_acid])
        self.current_vector = vec / np.linalg.norm(vec)

    def update_point(self, amino_acid):
        point = self.current_point
        self.update_vector(amino_acid)
        vec = self.current_vector
        self.current_point = point + vec

    def translate_dna(self):
        for acid in self.dna:
            self.update_point(acid)
            self.phenotype.append(self.current_point)

    def draw_phenotype(self):
        for

    @staticmethod
    def rotate_vector(xy, radians):
        """Use numpy to build a rotation matrix and take the dot product."""
        x, y = xy
        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])
        return np.array([float(m.T[0]), float(m.T[1])])




if __name__ == "__main__":
    worm = Vermiculus(1)
    print(worm.dna)
    worm.translate_dna()
    print(worm.phenotype)