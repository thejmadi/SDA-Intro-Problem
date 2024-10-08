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
    is_multi = True
    num_iter = 15
    
    # States in [r_x, r_y, v_x, v_y].T format

    # Create Instances of robots, 1 of type 1, 1 of type 2
    robots = []
    for k in range(4, 7):
        robots.append(ent.Robot(k))
        #robots.append(ent.Robot(k+1))
        #robots.append(ent.Robot(k+2))
        #robots.append(ent.Robot(k+3))
    robots[0].number_robots = len(robots)

    # Create Instances of sensors, 1 of type 1
    sensors = []
    for k in range(1):
        sensors.append(ent.Sensor(k))
    sensors[0].number_sensors = len(sensors)
    
    # Create Instance of Optimization()
    optimize = ent.Optimization()
    
    # Create Instance of RNG()
    if is_multi:
        rng_class = ent.RNG()
    else:
        rng_class = None
    
    # Outer-most loop/s; Optimizes Policy
    J_optimized = np.zeros((robots[0].number_robots, num_iter+1))
    J_optimized[:, 0] = fc.SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = False, rng_class = rng_class)
    plot.PlotRoom(robots)
    optimize.J.fill(0)
    for k in range(num_iter):
        print("Iter k = ", k+1)
        for t in range(optimize.T-1):
            print("Timestep: ", t+1)
            for n in range(optimize.N):
                optimize.FreezePolicy(n, t)
                fc.SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = True, rng_class = rng_class)
                #print(optimize.frozen_J)
                optimize.UpdatePartialJ(n, t)
        J_optimized[:, k+1] = fc.GradientDescent(robots, sensors, optimize, is_multi, rng_class = rng_class)
        optimize.Reset()
    '''
    if is_multi:
        J_optimized[0] = fc.SimulatePolicy(robots, sensors, optimize, is_frozen = False, rng_class)
        optimize.J.fill(0)
        for k in range(num_iter):
            print("Iter k = ", k+1)
            for t in range(optimize.T-1):
                print("Timestep: ", t+1)
                for n in range(optimize.N):
                    optimize.FreezePolicy(n, t)
                    fc.MultiMonteCarlo(robots, sensors, optimize, is_frozen = True)
                    optimize.UpdatePartialJ(n, t)
            J_optimized[k+1] = fc.GradientDescent(robots, sensors, optimize, is_multi, rng_class)
    
    else:
        J_optimized[0] = fc.MonteCarlo(robots, sensors, optimize, is_frozen = False)
        optimize.J.fill(0)
        for k in range(num_iter):
            print("Iter k =", k+1)
            for t in range(optimize.T-1):
                print("Timestep: ", t+1)
                for n in range(optimize.N):
                    optimize.FreezePolicy(n, t)
                    fc.MonteCarlo(robots, sensors, optimize, is_frozen = True)
                    optimize.UpdatePartialJ(n, t)
            J_optimized[k+1] = fc.GradientDescent(robots, sensors, optimize, is_multi)
            optimize.Reset()
    '''
    
    #cost = fc.MonteCarlo(robots, sensors, optimize, is_frozen = False)
    #print("Min cost: ")
    #print(min(J_optimized))
    # Call plots
    plot.OptimizedCost(num_iter, J_optimized)