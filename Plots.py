# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:40:56 2023

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

def PlotRoom(robots):
    
    # Plot visualization of room with robots' positions and estimates
    colors = ["blue", "green", "yellow"]
    plt.axvline(0, c='black', zorder=0)
    plt.axvline(robots[0].l[0], c='black', zorder=0)
    plt.axhline(0, c='black', zorder=0)
    plt.axhline(robots[0].l[1], c='black', zorder=0)
    for i in range(len(robots)):
        plt.scatter(robots[i].X_act[0, :], robots[i].X_act[1, :], c = colors[i], s = 40, label='Robot %i act' % (i+1))
        plt.scatter(robots[i].X_est[0, :], robots[i].X_est[1, :], c = 'r', s = 6, label = "Robot est")
    plt.legend(loc="best")
    plt.xlim(0 - 2, robots[0].l[0] + 2)
    plt.ylim(0 - 2, robots[0].l[1] + 2)
    plt.show()
    return

def PlotGraph(robots):
    colors = ["blue", "green", "yellow"]
    plots = ["r_x", "r_y", "v_x", "v_y"]
    
    # Plot robots' Est and Act States vs. Time
    for k in range(len(robots)):
        fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
        count = 0
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):    
                graphs[i][j].plot(robots[k].t, robots[k].X_act[count, :], c = colors[0], zorder=0, label = 'Act')
                graphs[i][j].scatter(robots[k].t, robots[k].X_est[count, :], c = colors[1], linestyle='--', label='Est')
                graphs[i][j].set_title(plots[count])
                count += 1
        fig.suptitle("Robot %i States vs. Time" % (k+1))
        fig.legend(loc="upper right")
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time")
        plt.show()
    
    # Plot est Error vs Time with error bars
    outside = 0
    total = 0
    for k in range(len(robots)):
        zero = np.zeros(robots[0].t.size)
        fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
        count = 0
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):    
                graphs[i][j].errorbar(robots[k].t, zero, yerr=robots[k].error_bars[count, :], fmt=' ', zorder=0)
                graphs[i][j].scatter(robots[k].t, robots[k].X_act[count, :] - robots[k].X_est[count, :], c = 'r', s =10)
                graphs[i][j].set_title(plots[count])
                if count <= 1:
                    for m in range(robots[0].t.size-1):
                        total += 1
                        if abs(robots[k].X_act[count, m] - robots[k].X_est[count, m]) >= robots[k].error_bars[count, m]:
                            outside += 1
                count += 1
        fig.suptitle("Robot %i Error with %i Sigma Error Bars vs. Time" % ((k+1), robots[k].sig_bounds))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time")
        plt.show()
        
    print(outside, "/", total, "estimations are outside of the errorbars (Position est only)")
    return