# Shapely_test_case
"""
a test case to determine whether or not shapely polygon patch calculates the
area that I expect when there is overlap.
Test cases are of interest
    case 1 - independant coords
        two unit coords with a half unit stack spaced 1.5 units apart
    case 2 - touching polygons
        two coords with half unit stack space 1 unit appart
    case 3 overlap
        two coords with half unit stack spaced half unit appart
    case 4 - crossing coords
        two perpendicular coords with half init stack
"""

import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString
from descartes.patch import PolygonPatch

def case1():
    """
    expected
    length
        2.0
    area
        1 + 1 + 2 * 0.7854 = 3.5708
    Returns
    -------

    """
    line1 = [(0, 0), (1, 0)]
    line2 = [(0, 1.5), (1, 1.5)]
    lines = [line1,line2]
    line = MultiLineString(lines)
    dilated = line.buffer(0.5)
    length = line.length
    area = dilated.area
    print("length : {0} \t area {1}:".format(length, area))
    return lines


def case2():
    """
        expected
        length
            2.0
        area
            1 + 1 + 2 * 0.7854 = 3.5708
        Returns
        -------

        """
    line1 = [(0, 0), (1, 0)]
    line2 = [(0, 1.0), (1, 1.0)]
    lines = [line1,line2]
    line = MultiLineString(lines)
    dilated = line.buffer(0.5)
    length = line.length
    area = dilated.area
    print("length : {0} \t area {1}:".format(length, area))
    return lines


def case3():
    """
        expected
        length
            2.0
        area
            2.762
        Returns
        -------

        """
    line1 = [(0, 0), (1, 0)]
    line2 = [(0, 0.5), (1, 0.5)]
    lines = [line1,line2]
    line = MultiLineString(lines)
    dilated = line.buffer(0.5)
    length = line.length
    area = dilated.area
    print("length : {0} \t area {1}:".format(length, area))
    return lines


def case4():
    """
    expected
    length
        2.0
    area
        1 + 2 * o.76 = 2.5682742
    Returns
    -------

    """
    line1 = [(-0.5, 0), (0.5, 0)]
    line2 = [(0, -0.5), (0, 0.5)]
    lines = [line1,line2]
    line = MultiLineString(lines)
    dilated = line.buffer(0.5)
    length = line.length
    area = dilated.area
    print("length : {0} \t area {1}:".format(length, area))
    return lines


def case5():
    """
    expected
    length
        2.0
    area
        1 + 0.76 = 1.76
    Returns
    -------

    """
    line1 = [(0, 0), (0, 1)]
    line2 = [(0, 0), (0, 1)]
    lines = [line1,line2]
    line = MultiLineString(lines)
    dilated = line.buffer(0.5)
    length = line.length
    area = dilated.area
    print("length : {0} \t area {1}:".format(length, area))
    return lines


def plot(lines):
    fig = plt.figure(1, figsize=(5, 5), dpi=180)
    ax = fig.add_subplot(111)
    line = MultiLineString(lines)

    dilated = line.buffer(0.5)
    patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
    ax.add_patch(patch1)
    for i in range(len(lines)):
        x, y = line[i].xy
        plt.axis('equal')
        ax.plot(x, y, color='#999999')
        plt.show()


case1()
case2()
case3()
case4()
case5()

plot(case5())