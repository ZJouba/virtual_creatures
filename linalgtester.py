import numpy as np

a = np.array((
    [1, 2, 3, 4],
    [2, 3, 4, 5]
))
b = np.array((
    [2, 4, 6, 8],
    [3, 5, 7, 9]
))

# lstsqrs = sum(np.linalg.norm(a-b, axis=0))

# print(lstsqrs)

print(sum(np.sqrt(((a-b)**2).sum(axis=0))))
