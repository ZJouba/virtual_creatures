import numpy as np
from timeit import default_timer as timer
from numba import vectorize

def pow(a, b, c):
    for i in range(a.size):
         c[i] = a[i] ** b[i]

@vectorize(['float32(float32, float32)'], target='cuda')
def pow_vec(a, b):
    return a ** b

def main():
    vec_size = 100000000

a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    pow(a, b, c)
    duration = timer() - start

    print(duration)

def main2():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = pow_vec(a, b)
    duration = timer() - start

    print(duration)

if __name__ == '__main__':
    main()
    main2()

#######################################################################
import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] = val
        return self.array

aaa = {"a": 2,
       "b": 3}

for key, value in aaa.items():
    print(key, value)




import numpy as np
from scipy.spatial.transform import Rotation
starting_point=np.array([0, 0])
starting_vector=np.array([0, 1])

r = Rotation.from_euler('z', -90, degrees=True)

vec = np.append(starting_vector, [0])

print(r.apply(vec))

def next_point(angle, vec, dist, point):
    r = Rotation.from_euler('z', angle, degrees=True)
    vec = np.append(vec, [0])
    vec = r.apply(vec)[:2]
    vec = dist * (vec / np.linalg.norm(vec))
    point = point + vec
    return point, vec


starting_point=np.array([0, 0])
starting_vector=np.array([0, 1])
print(next_point(90, starting_vector, 2, starting_point))




