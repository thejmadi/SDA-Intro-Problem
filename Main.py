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

if __name__ == '__main__':
    is_multi = False
    num_iter = 3
    J_optimized = np.zeros(num_iter)
    
    # States in [r_x, r_y, v_x, v_y].T format

    # Create Instances of robots, 1 of type 1, 1 of type 2
    robots = []
    for k in range(1):
        robots.append(ent.Robot(k))
        robots.append(ent.Robot(k+1))
    robots[0].number_robots = len(robots)

    # Create Instances of sensors, 1 of type 1
    sensors = []
    for k in range(1):
        sensors.append(ent.Sensor(k))
    sensors[0].number_sensors = len(sensors)
    
    # Create Instance of Optimization()
    optimize = ent.Optimization()

    # Create Instance of RNG()
    if is_multi: rng_class = ent.RNG()
    
    # Outer most loop - Optimizes Policy
    if is_multi:
        for k in range(num_iter):
            J_optimized[k] = fc.MultiMonteCarlo(robots, sensors, optimize, rng_class)
            
    else:
        for k in range(num_iter):
            print("Iter k =", k+1)
            J_optimized[k] = fc.MonteCarlo(robots, sensors, optimize, is_frozen = False)
            for t in range(optimize.T-1):
                print("Timestep: ", t+1)
                optimize.FreezePolicy(t)
                fc.MonteCarlo(robots, sensors, optimize, is_frozen = True)
                optimize.UpdatePartialJ(t)
                print()
            fc.GradientDescent(optimize)
            optimize.Reset()
            
    # Call plots
    plot.OptimizedCost(num_iter, J_optimized)