import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

x = np.linspace(0, 10, num=40)
y = 3.45 * np.sin(1.334 * x) + np.random.normal(size=40)


def test(x, a, b):
    return a * np.sin(b * x)


param, param_cov = curve_fit(test, x, y)

ans = (param[0]*(np.sin(param[1]*x)))

ax.plot(x, y, 'o', color='red', label="Data points")
ax.plot(x, ans, '--', color='blue', label="Best fit")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.legend()
plt.show()
