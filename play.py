import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})




fig = plt.figure(1, figsize=(5, 5), dpi=180)
ax = fig.add_subplot(111)
circ = Point(1,1).buffer(1)
circ2 = Point(2,2).buffer(1)

inter = circ.intersection(circ2)

patch1 = PolygonPatch(circ, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch1)
patch2 = PolygonPatch(circ2, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch2)
patch3 = PolygonPatch(inter, facecolor='red', edgecolor='#6699cc')
ax.add_patch(patch3)
# x, y = line.xy
plt.axis('equal')
# ax.plot(x, y, color='#999999')
plt.show()


fig = plt.figure(1, figsize=(5, 5), dpi=180)
ax = fig.add_subplot(111)
feed_zones = [(0, 0, 0.5), (1, 1, 0.5), (2, 2, 0.5)]
circs = [Point(zone[0], zone[1]).buffer(zone[2]) for zone in feed_zones]

patches = [PolygonPatch(circ) for circ in circs]
for patch in patches:
    ax.add_patch(patch)
plt.axis('equal')
plt.show()