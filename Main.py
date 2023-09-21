# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:39:35 2023

@author: tarun
"""

# Working Version
# Uses Entities.py to generate room with n robots and m sensors
# Uses Functions.py to calc actual and estimated states of robots using Kalman Filter
# Uses Plots.py to generate 


import cProfile
import pstats

import numpy as np
import Entities as ent
import Functions as fc
import Plots as plot

def Main():
    num_runs = 50
    # States in [r_x, r_y, v_x, v_y].T format

    # Initialize all robot instance initial conditions
    
    V_1 = np.array([2, 1])
    start_position_1 = np.array([5, 0])
    Q_1 = np.diag(np.array([0.25, 0.25, 0, 0]))
    
    V_2 = np.array([1, 2])
    start_position_2 = np.array([0, 5])
    Q_2 = np.diag(np.array([0.25, 0.25, 0, 0]))
    
    # Create all instances of robots
    robots = [ent.Robot(V_1, start_position_1, Q_1),
              ent.Robot(V_2, start_position_2, Q_2)]
    
    # Initialize all sensor instance initial conditions
    sensor_position_1 = np.array([0, 0])
    sensor_field_of_view_1 = np.array([[0, robots[0].l[0]/2], [0, robots[0].l[1]]])
    R_1 = np.diag(np.array([.25, .25]))
    
    # Create all instances of sensors
    sensors = [ent.Sensor(sensor_position_1, sensor_field_of_view_1, R_1)]
    
    # Call Kalman Filter
    robots, sensors = fc.MonteCarlo(robots, sensors, num_runs)
    
    # Call plots
    # Plot visualization of room with robots' positions and estimates, 1 plot
    #plot.PlotRoom(robots)
    # Plot robots' Est and Act States vs. Time and Plot est Error vs Time with error bars, 2 plots/robot
    #plot.PlotGraph(robots)
    #plot.PlotSensorTargets(sensors)
    
# Runs simulation specified number of times
#Main()
run = cProfile.Profile()
run.run("Main()")
run.dump_stats("cProfile.prof")

with open("cProfile.txt", "w") as txt:
    stats = pstats.Stats("cProfile.prof", stream = txt)
    stats.sort_stats("cumtime")
    stats.print_stats()