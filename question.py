from Tools.Classes import Limb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import numpy as np

limb = Limb()

orientation_vector = ["TOP", "BOTTOM", "TOP", "TOP",
                      "BOTTOM", "TOP", "BOTTOM", "TOP", "TOP", "BOTTOM", "TOP", "BOTTOM", "TOP", "TOP", "BOTTOM", ]

limb.build(orientation_vector)

segs = limb.XY.shape[1]

points = limb.XY

fig, ax = plt.subplots()
# ax.set_title("Soft actuator\n" + "Number of segments: {}".format(segs))
ax.plot([0, 0], [-2, 2], color='black')
ax.xaxis.set_major_locator(MultipleLocator(1))

"""------ACTUATED-------"""
ax.plot(points[0, :], points[1, :], color='red',
        label="Final pressure (P=P" + r'$_f$' + ")")

ax.set_aspect('equal', adjustable='datalim')

plt.show()
