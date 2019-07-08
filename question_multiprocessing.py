import multiprocessing as mp
from math import log
from functools import partial
import tqdm
import time

def func(x, y):
    return log(x % y)

if __name__ == "__main__":

    def_x = 0.5

    with mp.Pool(6) as pool:
        part_func = partial(func, def_x)
        """ WITH PROGRESSBARS """
        start = time.time()
        result = list(tqdm.tqdm(pool.imap(part_func, range(1,10)), total=2))
        end = time.time()
        print('{} s'.format(end - start))
        """ WITHOUT PROGRESSBARS """
        start = time.time()
        result = list(pool.imap(part_func, range(1,10)))
        end = time.time()
        print('{} s'.format(end - start))