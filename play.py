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