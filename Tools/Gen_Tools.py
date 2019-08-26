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
