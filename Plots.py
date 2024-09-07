# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:40:56 2023

@author: tarun
"""

# Working Version
# Plotting Functions - 4 functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
plt.rcParams['figure.dpi'] = 300

def PlotRoom(robots):
    colors = ["blue", "green", "yellow"]
    plt.axvline(0, c='black', zorder=0)
    plt.axvline(robots[0].l[0], c='black', zorder=0)
    plt.axvspan(0, 5, ymin=1/12, ymax=11/12, alpha=0.3, color='gray', label='Sensor FoV')
    plt.axhline(0, c='black', zorder=0)
    plt.axhline(robots[0].l[1], c='black', zorder=0)
    for i in range(len(robots)):
        plt.scatter(robots[i].X_act[0, :], robots[i].X_act[1, :], c = colors[i%3], s = 40, label='Robot %i act' % (i+1))
        #plt.scatter(robots[i].X_est[0, :], robots[i].X_est[1, :], c = 'r', s = 6, label = "Robot est")
    plt.legend(loc="best")
    plt.title("Room Plot")
    plt.xlim(0 - 1, robots[0].l[0] + 1)
    plt.ylim(0 - 1, robots[0].l[1] + 1)
    plt.show()
    # Plot visualization of room with robots' positions and estimates
    #colors = ["blue", "green", "yellow"]
    plt.axvline(0, c='black', zorder=0)
    plt.axvline(robots[0].l[0], c='black', zorder=0)
    plt.axvspan(0, 5, ymin=1/12, ymax=11/12, alpha=0.3, color='gray', label='Sensor FoV')
    plt.axhline(0, c='black', zorder=0)
    plt.axhline(robots[0].l[1], c='black', zorder=0)
    for i in range(len(robots)):
        plt.scatter(robots[i].X_act[0, :], robots[i].X_act[1, :], c = colors[i%3], s = 40, label='Robot %i act' % (i+1))
        plt.scatter(robots[i].X_est[0, :], robots[i].X_est[1, :], c = 'r', s = 6, label = "Robot %i est" % (i+1))
    plt.legend(loc="best")
    plt.title("Room Plot with Estimations")
    plt.xlim(0 - 1, robots[0].l[0] + 1)
    plt.ylim(0 - 1, robots[0].l[1] + 1)
    plt.show()
    return

def PlotGraph(robots):
    colors = ["blue", "green", "yellow"]
    plots = ["x Position", "y Position", "x Velocity", "y Velocity"]
    
    # Plot robots' Est and Act States vs. Time
    for k in range(len(robots)):
        fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
        count = 0
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):    
                graphs[i][j].plot(robots[k].t_array, robots[k].X_act[count, :], c = colors[0], zorder=0, label = 'Act')
                graphs[i][j].scatter(robots[k].t_array, robots[k].X_est[count, :], c = colors[1], linestyle='--', label='Est')
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
        zero = np.zeros(robots[0].T)
        fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
        count = 0
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):    
                graphs[i][j].errorbar(robots[k].t_array, zero, yerr=robots[k].error_bars[count, :], fmt=' ', zorder=0)
                graphs[i][j].scatter(robots[k].t_array, robots[k].X_act[count, :] - robots[k].X_est[count, :], c = 'r', s =10)
                graphs[i][j].set_title(plots[count])
                if count <= 1:
                    for m in range(robots[0].T-1):
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

def PlotSensorTargets(sensors):
    for k in range(len(sensors)):
        plt.scatter(sensors[k].t, sensors[k].targets_over_time + np.ones(sensors[k].t.shape[0]))
    plt.title("Sensor Target vs. Time")
    plt.ylabel("Robot Number")
    plt.xlabel("Time")
    plt.show()
    return

def OptimizedCost(runs, J):
    runs_arr = np.arange(0, runs+1, 1)
    for k in range(J.shape[0]):
        plt.plot(runs_arr, J[k, :], label="Robot " + str(k+1), linewidth=1.5)
    plt.title("Cost per Robot vs Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost")
    #print(runs_arr)
    #print(J)
    plt.plot(runs_arr, np.sum(J, axis=0), label="Total", linewidth=1.5)
    #plt.title("Total Cost vs Iteration")
    #plt.xlabel("Iteration Number")
    #plt.ylabel("Cost")
    plt.legend(loc="best")
    plt.show()
    plt.scatter(runs_arr, np.sum(J, axis=0))
    plt.yscale("log")
    plt.title("Log Cost vs Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost")
    plt.show()

def PlotHeatMapAnimation(data, num_frames, title, x_label, y_label, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(data[:, :, 0], cmap="gray")
    label = fig.text(0, 0, "Iter: 0", fontsize=10)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(im)
    im.set_clim(0, 1)
    def animate(i):
        im.set_data(data[:, :, i])
        label.set_text("Iter:" + str(i))
        return im

    anim = animation.FuncAnimation(fig, animate, frames=data.shape[2], repeat = False)

    plt.show()
    plt.rcParams['animation.ffmpeg_path'] = "D:\\ffmpeg-7.0.2-essentials_build\\ffmpeg-7.0.2-essentials_build\\bin\\ffmpeg.exe"
    writer = animation.FFMpegWriter(fps=0.25)
    anim.save(filename + '.mkv', writer=writer)