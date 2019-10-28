from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5])
A = np.array([1, 8, 12, 16, 20, 24])
B = np.array([1, 5, 21, 85, 341, 1365])
C = np.array([1, 9, 21, 45, 93, 189])

fig, ax = plt.subplots()

# ax.plot(A)
# ax.plot(B)
# ax.plot(C)

# plt.show()

xnew = np.linspace(x.min(), x.max(), 300)

Aspl = make_interp_spline(x, A, k=3)
Bspl = make_interp_spline(x, B, k=3)
Cspl = make_interp_spline(x, C, k=3)

Apower_smooth = Aspl(xnew)
Bpower_smooth = Bspl(xnew)
Cpower_smooth = Cspl(xnew)

plt.plot(xnew, Apower_smooth, label='FFFFX')
plt.plot(xnew, Bpower_smooth, label='XXXXF')
plt.plot(xnew, Cpower_smooth, label='FFFXX')

plt.title('Recursive property of L-strings')
plt.xlabel('Recursion')
plt.ylabel('Number of F characters in L-string')

plt.legend()

plt.show()
