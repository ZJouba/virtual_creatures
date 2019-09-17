import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from shapely import affinity


if __name__ == '__main__':

    x = np.array([[0.,   1.74729639,   3.49459277,   5.24188916,
                   6.73472277,   8.22755638,   9.72038999,  11.2132236,
                   12.70605721,  13.72702327,  14.74798933,  16.24082294,
                   17.73365655,  19.22649016,  20.71932377,  22.21215738],
                  [0.,   0.33964001,   0.67928002,   0.33964001,
                   -0.62981747,  -1.59927495,  -2.56873244,  -3.53818992,
                   -4.5076474,  -5.96573804,  -7.42382868,  -8.39328616,
                   -9.36274364, -10.33220112, -11.30165861, -12.27111609]])

    to_tuple = [(x, y) for x, y in zip(x[0], x[1])]
    line_check = LineString(to_tuple)

    new_line = affinity.translate(line_check, 5, 0)
    # new_line = affinity.rotate(new_line, 90)

    fig, ax = plt.subplots()
    ax.plot(line_check.xy[0], line_check.xy[1])
    ax.plot(new_line.xy[0], new_line.xy[1])
    plt.grid()
    plt.show()
