from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

root = tk.Tk()
root.attributes("-topmost", True)
root.withdraw()
filepath = filedialog.askopenfilename()

allData = pickle.load(open(filepath, 'rb'))

allData = allData['Area']

l = len(allData)
nums = [1000, 2000, 3000, 4000, 5000]
a = []
b = []
d = []
for num in nums:
    d.append(allData.sample(num))

colors = ['red', 'blue', 'green', 'yellow', 'purple']
names = ['1000 samples', '2000 samples',
         '3000 samples', '4000 samples', '5000 samples']

fig, ax = plt.subplots()

plt.hist(d, bins=1000, density=True, color=colors, label=names)
plt.xlim(1, 20)
plt.legend()
plt.xlabel('Absorption area')
plt.ylabel('Probability density')
# plt.show()
plt.savefig('C:\\Users\\zjmon\\Documents\\Meesters\\Thesis\\figs\\dist.pgf')
