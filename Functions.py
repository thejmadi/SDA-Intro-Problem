# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:47:08 2023

@author: tarun
"""

# Working Version
# Extra Functions
# Includes Error bar calculations, reset instances of robots/sensors, Kalman Filter, MonteCarlo

import sympy as sp
import numpy as np
from numpy import linalg as la
import Entities as ent
import Plots as plot
import multiprocessing as mlt
import matplotlib.pyplot as plt
#import sys

def ErrorBars(robot, k):
    robot.error_bars[:, k] = robot.sig_bounds * np.sqrt(np.diag(robot.P).astype(float))
    return

def ResetInstances(robots, sensors):
    for k in range(1):
        robots[k].Reset(k)
        robots[k+1].Reset(k+1)
    for k in range(1):
        sensors[k].Reset(k)

def KF(robots, sensors, optimize, is_perturbed, is_multi, rng_child = None):
    #rng_child.random()
    dim_state = robots[0].dim_state
    dim_msmt = robots[0].dim_msmt
    num_robots = len(robots)
    num_sensors = len(sensors)
    I = np.identity(dim_state)
    
    for k in range(sensors[0].t.size - 1):
        #print("k: ", k)
        # 1. Propagate
        for i in range(num_robots):
            ErrorBars(robots[i], k)
            # Propagate actual X
            robots[i].X_act[:, k+1] = robots[i].Propagation(robots[i].X_act[:, k], k+1, True, is_multi, rng_child)
            
            # Propagate est X; gives X_k+1 prior
            robots[i].X_est[:, k+1] = robots[i].Propagation(robots[i].X_est[:, k], k+1, False, is_multi, rng_child)
            # Due to difficulties with noncontinuous system ie. reflections off of walls,
            # Set velocities of est X to velocities of act X
            robots[i].X_est[2:, k+1] = robots[i].X_act[2:, k+1]
            
            # Propagate P; gives P_k+1 prior
            _, F, _, _, _, _, _ = robots[i].Dynamics(k)
            robots[i].P = (F @ robots[i].P) @ F.T + robots[i].G @ robots[i].w[1] @ robots[i].G.T
            
            ErrorBars(robots[i], k+1)
            F.fill(0)
        
        # 2a. Check if all actual X's is in sensors FoV
        for i in range(num_sensors):
            robot_choice = optimize.Tasking(k, is_perturbed, is_multi, rng_child)
            if sensors[i].InFoV(robots[robot_choice].X_act[:, k+1]):
                #print(k, robot_choice+1)
                sensors[i].SwitchTarget(robots[robot_choice], k+1, robot_choice)
            else:
                sensors[i].SwitchTarget(None, k+1, float("NaN"))
        
        # 2b. Update X, P of sensors' targets
        for i in range(num_sensors):
            if sensors[i].target != None:
                # Take observation data from actual X
                sensors[i].target.Y_act[:, k+1], _ = sensors[i].Obs(sensors[i].target.X_act[:, k+1], True, is_multi, rng_child)
                # Take observation data from actual X
                sensors[i].target.Y_est[:, k+1], H = sensors[i].Obs(sensors[i].target.X_est[:, k+1], False, is_multi, rng_child)
                
                # Remove bias from measurements
                Y_error = (sensors[i].target.Y_act[:, k+1] - sensors[i].target.Y_est[:, k+1]).reshape((dim_msmt, 1))

                # Calc K gain for each estimation
                sensors[i].K = sensors[i].target.P @ H.T @ la.pinv((H @ sensors[i].target.P) @ H.T + sensors[i].v[1])

                # Update est X
                sensors[i].target.X_est[:, k+1] = sensors[i].target.X_est[:, k+1] + (sensors[i].K @ (Y_error)).reshape(dim_state)

                # Update est P
                sensors[i].target.P = (I - sensors[i].K @ H) @ sensors[i].target.P

                ErrorBars(sensors[i].target, k+1)
                H.fill(0)
        
        # Compute CostPerRobot and CostPerTimestep
        for i in range(num_robots):
            optimize.CostPerRobot(robots[i].P, i)
        optimize.CostPerTimestep(k)

    return robots, sensors, optimize, 

def GradientDescent(optimize, J, J_pert):
    # Gradient Descent Formula
    gradient = (J_pert - J) / (optimize.perturbed_policy - optimize.current_policy)
    optimize.current_policy -= optimize.learn_rate * gradient

    # Fill optimize's perturbed policy with 0
    optimize.Reset3()
    
    # Enforce bounds for Policy elements
    for row in range(optimize.current_policy.shape[0]):
        for col in range(optimize.current_policy.shape[1]):
            if optimize.current_policy[row, col] > 1:
                optimize.current_policy[row, col] = 0.98
            elif optimize.current_policy[row, col] < 0:
                optimize.current_policy[row, col] = 0.02

def MultiTaskFunction(robots, sensors, optimize, is_perturbed, child_id):
    KF(robots, sensors, optimize, is_perturbed, True, child_id)
    # Calc cost per run
    optimize.MultiCostPerRun()

    return optimize.multi_J_runs

def MonteCarlo(num_runs):
    J = np.zeros(num_runs) # Unperturbed
    J_pert = np.zeros(num_runs) # Perturbed
    robots = []
    
    # Create Instances of robots, 1 of type 1, 1 of type 2
    for k in range(1):
        robots.append(ent.Robot(k))
        robots.append(ent.Robot(k+1))
    robots[0].number_robots = len(robots)

    sensors = []

    # Create Instances of sensors, 1 of type 1
    for k in range(1):
        sensors.append(ent.Sensor(k))
    sensors[0].number_sensors = len(sensors)
    
    # Create Instance of Optimization()
    optimize = ent.Optimization(len(robots))
    
    # Run Monte Carlo (num_runs * MC_runs) times
    for r in range(num_runs):
        # Start: Monte Carlo for Unperturbed Policy_k
        for n in range(optimize.MC_runs):
            # Call Kalman Filter Function
            robots, sensors, optimize = KF(robots, sensors, optimize, False, False)
            # Calc cost per run
            optimize.CostPerRun(n)
            #plot.PlotRoom(robots)
            # Reset optimize, robots, sensors
            optimize.Reset1()
            ResetInstances(robots, sensors)
        # Calc Total Cost for 1 Monte Carlo batch
        optimize.CostTotal()
        J[r] = optimize.J
        plot.PlotRoom(robots)
        if J[r] < optimize.min_cost:
            optimize.min_cost = J[r]
            optimize.optimal_policy = optimize.current_policy
            optimize.min_iter = r
        optimize.Reset2()
        # End: Monte Carlo for Unperturbed Policy_k 

        # Start: Monte Carlo for Perturbed Policy_k
        optimize.PerturbPolicy()
        
        for n in range(optimize.MC_runs):
            # Call Kalman Filter Function
            robots, sensors, optimize = KF(robots, sensors, optimize, True, False)
            # Calc cost per run
            optimize.CostPerRun(n)
            
            # Reset optimize, robots, sensors
            optimize.Reset1()
            ResetInstances(robots, sensors)
        
        # Calc Total Cost for 1 Monte Carlo batch
        optimize.CostTotal()
        #optimize.ToFile()

        J_pert[r] = optimize.J
        optimize.Reset2()
        # End: Monte Carlo for Perturbed Policy_k
        
        # Start: Gradient Descent
        GradientDescent(optimize, J[r], J_pert[r])
        # End: Gradient Descent
        print("Iter: ", r+1, "Cost: ", J[r])
        print(optimize.current_policy)
        print()

    print("Min Iter:", optimize.min_iter)
    print("Min Cost: ", optimize.min_cost)
    print("Optimal Pol: ", optimize.optimal_policy)
    return J

# Uses Multiprocessing
def MultiMonteCarlo(num_runs):
    J = np.zeros(num_runs)
    J_pert = np.zeros(num_runs)
    robots = []
    
    # Create Instances of robots, 1 of type 1, 1 of type 2
    for k in range(1):
        robots.append(ent.Robot(k))
        robots.append(ent.Robot(k+1))
    robots[0].number_robots = len(robots)

    sensors = []

    # Create Instances of sensors, 1 of type 1
    for k in range(1):
        sensors.append(ent.Sensor(k))
    sensors[0].number_sensors = len(sensors)
    
    # Create Instance of Optimization()
    optimize = ent.Optimization(len(robots))

    # Create Instance of RNG()
    rng_class = ent.RNG()
    
    # Run Monte Carlo (num_runs * MC_runs) times
    for r in range(num_runs):        
        # Start: Monte Carlo for Unperturbed Policy_k
        with mlt.Pool(6) as pool:
            multi_results = [pool.apply_async(MultiTaskFunction, args=(robots, sensors, optimize, False, rng_class.rng_children[child_id])) for child_id in range(optimize.MC_runs)]
            J_runs = [r.get() for r in multi_results]
        # Calc Total Cost for 1 Monte Carlo batch
        J[r] = np.sum(J_runs)/optimize.MC_runs
        
        if J[r] < optimize.min_cost:
            optimize.min_cost = J[r]
            optimize.optimal_policy = optimize.current_policy
            optimize.min_iter = r
        # End: Monte Carlo for Unperturbed Policy_k
        
        # Start: Monte Carlo for Perturbed Policy_k
        optimize.PerturbPolicy(True, rng_class.rng_children[-1])
        
        with mlt.Pool(6) as pool:
            multi_results = [pool.apply_async(MultiTaskFunction, args=(robots, sensors, optimize, True, rng_class.rng_children[child_id])) for child_id in range(optimize.MC_runs)]
            J_runs = [r.get() for r in multi_results]
        J_pert[r] = np.sum(J_runs)/optimize.MC_runs
        # End: Monte Carlo for Perturbed Policy_k
        
        # Start: Gradient Descent
        GradientDescent(optimize, J[r], J_pert[r])
        # End: Gradient Descent
        print("Iter: ", r+1, "Cost: ", J[r])
        print(optimize.current_policy)
        print()

    print("Min Iter:", optimize.min_iter)
    print("Min Cost: ", optimize.min_cost)
    print("Optimal Pol: ", optimize.optimal_policy)
    return J