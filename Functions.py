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

def KF(robots, sensors, optimize, is_multi, is_frozen, rng_child = None):
    #rng_child.random()
    dim_state = robots[0].dim_state
    dim_msmt = robots[0].dim_msmt
    num_robots = len(robots)
    num_sensors = len(sensors)
    I = np.identity(dim_state)
    
    for t in range(sensors[0].T - 1):
        #print("k: ", k)
        # 1. Propagate
        for n in range(num_robots):
            ErrorBars(robots[n], t)
            # Propagate actual X
            robots[n].X_act[:, t+1] = robots[n].Propagation(robots[n].X_act[:, t], t+1, True, is_multi, rng_child)
            
            # Propagate est X; gives X_k+1 prior
            robots[n].X_est[:, t+1] = robots[n].Propagation(robots[n].X_est[:, t], t+1, False, is_multi, rng_child)
            # Due to difficulties with noncontinuous system ie. reflections off of walls,
            # Set velocities of est X to velocities of act X
            robots[n].X_est[2:, t+1] = robots[n].X_act[2:, t+1]
            
            # Propagate P; gives P_k+1 prior
            _, F, _, _, _, _, _ = robots[n].Dynamics(t)
            robots[n].P = (F @ robots[n].P) @ F.T + robots[n].G @ robots[n].w[1] @ robots[n].G.T
            
            ErrorBars(robots[n], t+1)
            F.fill(0)
        
        # 2a. Check if all actual X's is in sensors FoV
        for s in range(num_sensors):
            robot_choice = optimize.Tasking(t, is_multi, is_frozen, rng_child)
            if sensors[s].InFoV(robots[robot_choice].X_act[:, t+1]):
                #print(k, robot_choice+1)
                sensors[s].SwitchTarget(robots[robot_choice], t+1, robot_choice)
            else:
                sensors[s].SwitchTarget(None, t+1, float("NaN"))
        
        # 2b. Update X, P of sensors' targets
        for s in range(num_sensors):
            if sensors[s].target != None:
                # Take observation data from actual X
                sensors[s].target.Y_act[:, t+1], _ = sensors[s].Obs(sensors[s].target.X_act[:, t+1], True, is_multi, rng_child)
                # Take observation data from actual X
                sensors[s].target.Y_est[:, t+1], H = sensors[s].Obs(sensors[s].target.X_est[:, t+1], False, is_multi, rng_child)
                
                # Remove bias from measurements
                Y_error = (sensors[s].target.Y_act[:, t+1] - sensors[s].target.Y_est[:, t+1]).reshape((dim_msmt, 1))

                # Calc K gain for each estimation
                sensors[s].K = sensors[s].target.P @ H.T @ la.pinv((H @ sensors[s].target.P) @ H.T + sensors[s].v[1])

                # Update est X
                sensors[s].target.X_est[:, t+1] = sensors[s].target.X_est[:, t+1] + (sensors[s].K @ (Y_error)).reshape(dim_state)

                # Update est P
                sensors[s].target.P = (I - sensors[s].K @ H) @ sensors[s].target.P

                ErrorBars(sensors[s].target, t+1)
                H.fill(0)
        
        # Compute J
        for n in range(num_robots):
            optimize.UpdateJ(robots[n].P, n, t, is_frozen)

    return None

def GradientDescent(optimize):
    # Gradient Descent Formula
    #print(optimize.partial_J)
    for t in range(optimize.T-1):
        for n in range(1, optimize.N):
            optimize.partial_J[n, t] = optimize.J[n, t] - optimize.J[0, t] + optimize.partial_J[0, t]
    print(optimize.J)
    print(optimize.partial_J)
    optimize.current_policy -= optimize.learn_rate * optimize.partial_J
    
    # Enforce bounds for Policy elements
    for row in range(optimize.current_policy.shape[0]):
        for col in range(optimize.current_policy.shape[1]):
            if optimize.current_policy[row, col] > 1:
                optimize.current_policy[row, col] = 0.98
            elif optimize.current_policy[row, col] < 0:
                optimize.current_policy[row, col] = 0.02
    print(optimize.current_policy)
    return None

def MonteCarlo(robots, sensors, optimize, is_frozen):
    for n in range(optimize.MC_runs):
        KF(robots, sensors, optimize, False, is_frozen)
        ResetInstances(robots, sensors)
    return np.sum(optimize.J)

def MultiTaskFunction(robots, sensors, optimize, is_perturbed, child_id):
    KF(robots, sensors, optimize, is_perturbed, True, child_id)
    # Calc cost per run
    optimize.MultiCostPerRun()

    return optimize.multi_J_runs

# Uses Multiprocessing
def MultiMonteCarlo(num_runs):
    
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