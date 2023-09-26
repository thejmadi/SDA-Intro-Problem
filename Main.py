# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:39:35 2023

@author: tarun
"""

# Working Version
# Uses Entities.py to generate room with n robots and m sensors and 1 optimize instance
# Uses Functions.py to calc actual and estimated states of robots using Kalman Filter
# Uses Plots.py to generate 


import cProfile
import pstats

import numpy as np
import Entities as ent
import Functions as fc
import Plots as plot

def Main():
    num_runs = 20
    J_optimized = np.zeros(num_runs)
    
    # States in [r_x, r_y, v_x, v_y].T format
    
    # Call Kalman Filter
    J_optimized = fc.MonteCarlo(num_runs)
    
    # Call plots
    plot.OptimizedCost(J_optimized)
    # Plot visualization of room with robots' positions and estimates, 1 plot
    #plot.PlotRoom(robots)
    # Plot robots' Est and Act States vs. Time and Plot est Error vs Time with error bars, 2 plots/robot
    #plot.PlotGraph(robots)
    #plot.PlotSensorTargets(sensors)
    
run = cProfile.Profile()
run.run("Main()")
run.dump_stats("cProfile.prof")

with open("cProfile.txt", "w") as txt:
    stats = pstats.Stats("cProfile.prof", stream = txt)
    stats.sort_stats("cumtime")
    stats.print_stats()