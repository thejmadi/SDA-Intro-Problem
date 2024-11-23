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
import time

import numpy as np
import Entities as ent
import Functions as fc
import Plots as plot

if __name__ == '__main__':
    is_multi = True
    num_iter = 3
    
    # States in [r_x, r_y, v_x, v_y].T format

    # Create Instances of robots, 1 of type 1, 1 of type 2
    robots = []
    for k in range(10):
        robots.append(ent.Robot(k))
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
    
    mcmc = ent.MCMC()
    '''
    fc.KF(robots, sensors, optimize, False, False, rng_class, optimize.current_policy)
    #plot.PlotGraph(robots)
    plot.PlotRoom(robots, "Robot Trajectories")
    robots[0].Q_robot.fill(0)
    fc.ResetInstances(robots, sensors)
    optimize.Reset()

    fc.KF(robots, sensors, optimize, False, False, rng_class, optimize.current_policy)
    plot.PlotRoom(robots, "Noiseless Trajectories")
    fc.ResetInstances(robots, sensors)
    optimize.Reset()
    '''
    '''
    fov_stochastic = np.zeros((optimize.N, optimize.T-1))
    for k in range(1000):
        fc.KF(robots, sensors, optimize, False, False, rng_class, optimize.current_policy)
        fov_stochastic[:, :] += sensors[0].robots_in_FoV[:, :]
        fc.ResetInstances(robots, sensors)
        optimize.Reset()
    optimize.Q_robot.fill(0)
    optimize.R_sensor.fill(0)
    plot.PlotMCMCHist(fov_stochastic, "Stochastic FoV Heat Map", "Time t", "Robot n")
    #fov_deterministic = np.zeros((optimize.N, optimize.T-1))
    #fc.KF(robots, sensors, optimize, False, False, rng_class, optimize.current_policy)
    #fov_deterministic[:, :] += sensors[0].robots_in_FoV[:, :]
    #plot.PlotMCMCHist(fov_deterministic, "Deterministic FoV Heat Map", "Time t", "Robot n")
    '''
    mcmc_samples = mcmc.ParallelizeChains(robots, sensors, optimize, rng_class)
    burn_num = len(mcmc_samples[0][2])
    costs = mcmc_samples[0][0]
    min_cost = np.min(np.sum(costs, axis=0))
    min_idx = np.where(np.sum(costs,axis=0)==min_cost)
    samples_totals = mcmc_samples[0][6]
    samples_totals[0] = burn_num
    
    plot.OptimizedCost(costs.shape[1]-1, costs, "Burn and Sample Cost vs Iteration")
    #plot.PlotHeatMapAnimation(optimize.all_costs, 1, "Cost", "Time t", "Robot n", "cost_11_11_24_robots_4_5", is_cost=True)
    
    #aggregate = np.zeros((optimize.N, optimize.T-1))
    
    #plot.OptimizedCost(mcmc_samples[1].shape[1]-1, mcmc_samples[1], "Cost per Robot vs Iteration Chain")
    #plot.PlotHeatMapAnimation(mcmc_samples[0], mcmc_samples[0].shape[2], "Updated Policy Chain", "Time t", "Robot n", "MCMC_burn_check_1")
    #print()
    #for m in range(mcmc.num_processes):
        #plot.PlotHeatMapAnimation(mcmc_samples[m][0], mcmc.num_samples, "Updated Policy Chain #"+str(m+1), "Time t", "Robot n", "MCMC_spaced_samples_1"+str(m+1))
        #plot.PlotMCMCHist(np.sum(mcmc_samples[m][0], axis=2)/mcmc.num_samples, "Policy Histogram for MCMC Chain #" + str(m+1), "Time t", "Robot n")
        #plot.OptimizedCost(mcmc.num_burn + mcmc.num_samples - 1, mcmc_samples[m][1], "Cost per Robot vs Iteration Chain #" + str(m+1))
        #aggregate[:, :] += np.sum(mcmc_samples[m][0], axis=2)
    #aggregate /= mcmc.num_chains*mcmc.num_samples
    #plot.PlotMCMCHist(aggregate, "Aggregate Policy Histogram for MCMC", "Time t", "Robot n")
    #print(np.sum(aggregate, axis=1))
    
    
    #fc.KF(robots, sensors, optimize, True, False, rng_class.rng_children[-1])
    #plot.PlotGraph(robots)
    #fc.ResetInstances(robots, sensors)
    #optimize.Reset()
    
    # Outer-most loop/s; Optimizes Policy
    #J_optimized = np.zeros((robots[0].number_robots, num_iter+1))
    #J_optimized[:, 0] = fc.SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = False, rng_class = rng_class)

    #plot.PlotRoom(robots)
    #optimize.J.fill(0)
    '''
    for k in range(num_iter):
        print("Iter k = ", k+1)
        for t in range(optimize.T-1):
            #print("Timestep: ", t+1)
            for n in range(optimize.N):
                optimize.FreezePolicy(n, t)
                fc.SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = True, rng_class = rng_class)
                #print(optimize.frozen_J)
                optimize.UpdatePartialJ(n, t)
        J_optimized[:, k+1] = fc.GradientDescent(k+1, robots, sensors, optimize, is_multi, rng_class = rng_class)
        optimize.learn_rate *= 0.9
        optimize.all_policies[:, :, k+1] = optimize.current_policy
        optimize.Reset()
    '''
    '''Deprecated Code
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
    #plot.PlotHeatMapAnimation(optimize.all_policies, num_iter, "Updated Policy", "Time t", "Robot n", "IRHCPolicyUpdate3")
    #plot.OptimizedCost(num_iter, J_optimized, "Cost per Robot vs Iteration")