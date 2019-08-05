import random
import re
import time
import timeit

method1 = """
def method1():
    rules = {
        'X': ['XXXXF', 'XXXXF']
    }


    L_string = 'FX'


    def next_char(c):
        if c not in rules:
            return c
        d = rules[c]
        r = int(random.random() * len(d))
        return d[r]


    def re_next_char(c):
        return random.choice(rules[c.group()])


    for _ in range(6):
        L_string = ''.join([next_char(c) for c in L_string])"""


# start = time.time()
# for _ in range(6):
#     L_string = re.sub('|'.join(rules.keys()), re_next_char, L_string)
# end = time.time()

# print('Second method = {:5f}s'.format(end - start))

# result = []
# write = result.append


# def X(level):
#     if level == 0:
#         write('X')
#         return
#     if random.randint(0, 1):
#         # X -> XXXXF
#         X(level-1)
#         X(level-1)
#         X(level-1)
#         X(level-1)
#         write('F')
#     else:
#         # X -> XXXXF
#         X(level-1)
#         X(level-1)
#         X(level-1)
#         X(level-1)
#         write('F')


# def begin():
#     write('F')
#     X(5)  # 5 = recursion depth


# start = time.time()
# begin()
# L_string = ''.join(result)
# end = time.time()
method2 = """
def method2():
    rules = {
        'X': lambda: random.choice(["{X}{X}{X}{X}{F}", "{X}{X}{X}{X}{F}"]),
        'F': lambda: 'F',
    }


    def get_callbacks():
        while True:
            yield {k: v() for k, v in rules.items()}


    callbacks = get_callbacks()
    L_string = "{F}{X}"
    for _ in range(5):
        L_string = L_string.format(**next(callbacks))

    L_string = re.sub('{|}', '', L_string)"""

print('First method = \t\t{:.10f}'.format(timeit.timeit(
    stmt=method1,
    number=10000
)/10000))

print('Second method = \t{:.10f}'.format(timeit.timeit(
    stmt=method2,
    number=10000
)/10000))
