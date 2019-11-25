# -*- coding: utf-8 -*-
from matplotlib import cbook
import numpy as np
import matplotlib.pyplot as plt
"""
Created on Fri Jun 22 12:01:11 2018

@author: David
"""


def plotDavid(ax):

    import os

    directory = "C:\\Users\\zjmon\\Documents\\Meesters\\Ellis"  # os.getcwd()

    def punchToArray(filename, directory):
        with open('%s%s' % (directory, filename), "r") as f:
            searchlines = f.readlines()

        inc = 1  # Initialize a counter that will be used to step through increments number 1-9
        # Initialize a vector to step through all the nodes of importance (Node info in PCH are 2 lines apart)
        nodes = np.linspace(1, QUnits*8-1, QUnits*4, dtype=int)
        # Initialize empty vector that will be filled with x-displacments from PCH file (10 increments of 4 nodes)
        x = np.empty(4*QUnits*10)
        # Initialize empty vector that will be filled with y-displacments from PCH file (10 increments of 4 nodes)
        y = np.empty(4*QUnits*10)

        for i, line in enumerate(searchlines):
            # This line investigates all the first 9 increments in the solution
            if ('$TIME =   0.%s000000E+00' % (inc)) in line:
                for node in nodes:
                    x[int((inc-1)*60+(node+1)/2-1)
                      ] = searchlines[i+node][23:36]
                    y[int((inc-1)*60+(node+1)/2-1)
                      ] = searchlines[i+node][41:54]
                inc = inc + 1

            if ('$TIME =   0.1000000E+01') in line:
                for node in nodes:
                    x[int((inc-1)*60+(node+1)/2-1)
                      ] = searchlines[i+node][23:36]
                    y[int((inc-1)*60+(node+1)/2-1)
                      ] = searchlines[i+node][41:54]

        return x, y

    def profileSmooth(xNodal):
        y = []
        y = (10*np.cos((xNodal)/(max(xNodal)/(2*np.pi)))-10)
        return y

    def profile(xNodal):
        y = []
        xInitial = np.linspace(5, 10.5*QUnits-5.5, QUnits)
        for count, entry in enumerate(xNodal, start=0):
            y.append(
                40*np.cos((entry+xInitial[count])/(max(xInitial+xNodal)/(2*np.pi)))-40)
        return y, xInitial

    QUnits = 15
    x2temp, y2temp = punchToArray(
        '\\cos20_pc_2d_sim1-solution_1.pch', directory)
    x3temp, y3temp = punchToArray(
        '\\cos20_manual_3d_sim1-solution_1_nodes.pch', directory)

    x2 = []
    for count, entry in enumerate(x2temp):
        if ((count+1) % 4) == False:
            x2.append(sum(x2temp[count-3:count+1])/4)

    y2 = []
    for count, entry in enumerate(y2temp):
        if ((count+1) % 4) == False:
            y2.append(sum(y2temp[count-3:count+1])/4)

    x3 = []
    for count, entry in enumerate(x3temp):
        if ((count+1) % 4) == False:
            x3.append(sum(x3temp[count-3:count+1])/4)

    y3 = []
    for count, entry in enumerate(y3temp):
        if ((count+1) % 4) == False:
            y3.append(sum(y3temp[count-3:count+1])/4)

    Inc = 10

    x2Smooth = np.linspace(0, 157.5 - 5.5 + x2[14 + 15*(Inc-1)], 51)
    yProfileSmooth = profileSmooth(x2Smooth)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    xInitial = np.linspace(5, 10.5*QUnits-5.5, QUnits)

    ax.plot(x3[((Inc-1)*15):(15+(Inc-1)*15)] + xInitial, y3[((Inc-1)*15):(15+(Inc-1)*15)],
            label='Ellis - Nodal disp. 3D', color='b', linestyle='-.', linewidth=3)
