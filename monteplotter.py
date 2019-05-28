import hvplot
import pandas as pd
import time
import dask.dataframe as dd
import itertools
import numpy as np
from tkinter import filedialog
import tkinter as tk
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def coordLengths(coords):
    (p1, p2) = zip(*[itertools.islice(coords, i, None, 2) for i in range(2)])
    length = np.linalg.norm((p1, p2))
    return length


root = tk.Tk()
root.withdraw()
curDir = os.path.dirname(__file__)

filepath = filedialog.askopenfilename()
filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]

monteData = dd.read_csv(filepath)

monteData.fillna(0)

pairData = monteData.sample(frac=0.1)
# monteData['Length'] = monteData['Bounding Coordinates'].apply(
#     coordLengths)
# monteData.reset_index(drop=True)
# monteData.style.format(
#     {'% of F': '{:.2%}', '% of +': '{:.2%}', '% of -': '{:.2%}'})

# print(monteData.head())
hvplot.scatter_matrix(pairData, c='Area')
# sns.pairplot(monteData, hue='Area')
plt.show()
