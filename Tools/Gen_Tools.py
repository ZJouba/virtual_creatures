def open_file():
    from tkinter import filedialog
    import tkinter as tk
    import pickle

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filepath = filedialog.askopenfilename()

    if filepath:
        print('\nFile selected')
        return pickle.load(open(filepath, 'rb'))
    else:
        exit()


def plot_creature(linestring):
    import matplotlib.pyplot as plt
    from descartes.patch import PolygonPatch
    from shapely.geometry import LineString

    fig, ax = plt.subplots()

    linestring_patch = PolygonPatch(
        linestring.buffer(0.5), fc='BLACK', alpha=0.1)

    ax.add_patch(linestring_patch)

    for line in linestring:
        x, y = line.xy
        ax.plot(x, y, 'r-')

    # for m in all_lines:
    #     for line in m:
    #         x, y = line.xy
    #         ax.plot(x, y, 'g--', alpha=0.25)

    ax.axis('equal')
    plt.show()


def overlay_images(ax, limb):
    import numpy as np
    from matplotlib import transforms
    from math import cos, radians, degrees
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    def imshow_affine(ax, z, *args, **kwargs):
        im = ax.imshow(z, *args, **kwargs)
        _, x2, y1, _ = im.get_extent()
        im._image_skew_coordinate = (x2, y1)
        return im

    segs = len(limb.orient)
    rotations = limb.curvature
    width = 25.7
    height = 78

    image_directory = "C:\\Users\\zjmon\\Documents\\Meesters\\virtual_creatures\\box.PNG"
    img = mpimg.imread(image_directory, format='png')

    cps = [[], []]
    for i in range(segs):
        cps[0].append((limb.XY[0][i] + limb.XY[0][i+1])/2)
        cps[1].append((limb.XY[1][i] + limb.XY[1][i+1])/2)
    cps = np.asarray(cps)

    for i in range(cps.shape[1]):
        img_show = imshow_affine(
            ax,
            img,
            interpolation='none',
            extent=[0, width, 0, height],
            alpha=0.6
        )

        c_x, c_y = width/2, (17.5*cos(radians(11)))

        if limb.orient[i] == "TOP":
            rot_angle = 180 + degrees(rotations[i+1])
        elif limb.orient[i] == "BOTTOM":
            rot_angle = degrees(rotations[i+1])
        else:
            rot_angle = degrees(rotations[i+1])

        transform_data = (transforms.Affine2D()
                          .rotate_deg_around(c_x, c_y, rot_angle)
                          .translate((cps[0][i]-c_x), (cps[1][i]-c_y))
                          + ax.transData)

        img_show.set_transform(transform_data)


def delete_lines(n=1):
    import sys

    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
