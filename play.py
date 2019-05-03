from matplotlib import pyplot
from shapely.geometry import LineString
from descartes.patch import PolygonPatch

fig = pyplot.figure(1, figsize=(10, 4), dpi=180)

# Plot 1: dilating a line
line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
ax = fig.add_subplot(121)
dilated = line.buffer(1.0)
patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch1)
x, y = line.xy
ax.plot(x, y, color='#999999')
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)

# Plot 2: eroding the polygon from 1
ax = fig.add_subplot(122)
patch2a = PolygonPatch(dilated, facecolor='#cccccc', edgecolor='#999999')
ax.add_patch(patch2a)
eroded = dilated.buffer(-0.3)
patch2b = PolygonPatch(eroded, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch2b)
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)

pyplot.show()
