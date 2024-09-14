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
import copy
#import sys


def ErrorBars(robot, k):
    robot.error_bars[:, k] = robot.sig_bounds * np.sqrt(np.diag(robot.P).astype(float))
    return

def ResetInstances(robots, sensors):
    for k in range(3):
        robots[k].Reset(k)
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

def SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen, rng_class):
    if is_multi:
        J_policy = MultiMonteCarlo(robots, sensors, optimize, is_frozen, rng_class)
    else:
        J_policy = MonteCarlo(robots, sensors, optimize, is_frozen)
    optimize.J.fill(0)
    return J_policy
    
def GradientDescent(iter_num, robots, sensors, optimize, is_multi, rng_class):
    # Gradient Descent Formula
    #print(optimize.partial_J)
    #print("Frozen Expected Costs: ")
    #print(optimize.partial_J)
    #print()
    #if iter_num == 5:
        #print()
    for t in range(optimize.T-1):
        for n in range(optimize.N-1):
            optimize.partial_J[n, t] = optimize.partial_J[n, t] - optimize.partial_J[-1, t]
    
    J_curr = SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = False, rng_class = rng_class)
    temp_policy = np.zeros(optimize.current_policy.shape)
    temp_policy[:, :] = optimize.current_policy[:, :]
    optimize.current_policy[:-1, :] -= optimize.learn_rate * optimize.partial_J[:-1, :]
    
    # Enforce bounds for rows 0 -> N-1
    for row in range(optimize.current_policy.shape[0]-1):
        for col in range(optimize.current_policy.shape[1]):
            #if optimize.current_policy[row, col] > 1:
            #    optimize.current_policy[row, col] = 1.0
            if optimize.current_policy[row, col] < 0:
                optimize.current_policy[row, col] = 0.0
    
    optimize.current_policy[-1, :] = np.maximum(0.0, 1 - np.sum(optimize.current_policy[:-1, :], axis=0))
    optimize.current_policy = optimize.current_policy / np.sum(optimize.current_policy, axis=0)
    
    J_new = SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = False, rng_class = rng_class)
    
    optimize.J.fill(0)
    
    print("J_curr = ", np.sum(J_curr))
    print("J_new = ", np.sum(J_new))
    
    if np.sum(J_new) > np.sum(J_curr):    
        optimize.learn_rate *= 0.8
        optimize.current_policy[:-1, :] = temp_policy[:-1, :] - optimize.learn_rate * optimize.partial_J[:-1, :]
        for row in range(optimize.current_policy.shape[0]-1):
            for col in range(optimize.current_policy.shape[1]):
                #if optimize.current_policy[row, col] > 1:
                #    optimize.current_policy[row, col] = 1.0
                if optimize.current_policy[row, col] < 0:
                    optimize.current_policy[row, col] = 0.0
        
        optimize.current_policy[-1, :] = np.maximum(0.0, 1 - np.sum(optimize.current_policy[:-1, :], axis=0))
        optimize.current_policy = optimize.current_policy / np.sum(optimize.current_policy, axis=0)
        
        J_new = SimulatePolicy(robots, sensors, optimize, is_multi, is_frozen = False, rng_class = rng_class)
        
        print("J_new = ", np.sum(J_new))
        optimize.J.fill(0)
    #J_new = J_curr
    print("Updated Policy: ")
    print(optimize.current_policy)
    #print("Sum: ")
    #print(np.sum(optimize.current_policy, axis=0))
    print()
    return J_new

def MonteCarlo(robots, sensors, optimize, is_frozen):
    for n in range(optimize.MC_runs):
        KF(robots, sensors, optimize, False, is_frozen)
        ResetInstances(robots, sensors)
    return np.sum(optimize.J)

def MultiTaskFunction(robots, sensors, optimize, is_frozen, child_id):
    KF(robots, sensors, optimize, True, is_frozen, child_id)
    # Calc cost per run
    if is_frozen:
        J = optimize.frozen_J
    else:
        J = optimize.J
    
    return J

# Uses Multiprocessing
def MultiMonteCarlo(robots, sensors, optimize, is_frozen, rng_class):
    J_run = np.zeros((optimize.N, optimize.T-1))
    with mlt.Pool(6) as pool:
        multi_results = [pool.apply_async(MultiTaskFunction, args=(robots, sensors, optimize, is_frozen, rng_class.rng_children[child_id])) for child_id in range(optimize.MC_runs)]
        for r in multi_results:
            J_run += r.get()
    # Calc Total Cost for 1 Monte Carlo batch
    if is_frozen:
        optimize.frozen_J = copy.deepcopy(J_run)
    else:
        optimize.J = copy.deepcopy(J_run)
    return np.sum(optimize.J, axis=1)