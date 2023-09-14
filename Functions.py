# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:47:08 2023

@author: tarun
"""

# Working Version
# Extra Functions
# Includes Error bar calculations, Kalman Filter, will include sensor tasking functions next

import sympy as sp
import numpy as np
from numpy import linalg as la

def ErrorBars(robot, k):
    robot.error_bars[:, k] = robot.sig_bounds * np.sqrt(np.diag(robot.P).astype(float))
    return

def KF(robots, sensors):
    dim_state = robots[0].dim_state
    dim_msmt = robots[0].dim_msmt
    num_robots = len(robots)
    num_sensors = len(sensors)
    I = np.identity(dim_state)
    
    for k in range(sensors[0].t.size - 1):
        # 1. Propagate
        for i in range(num_robots):
            ErrorBars(robots[i], k)
            # Propagate actual X
            robots[i].X_act[:, k+1] = robots[i].Propagation(robots[i].X_act[:, k], k+1, True)
            
            # Propagate est X; gives X_k+1 prior
            robots[i].X_est[:, k+1] = robots[i].Propagation(robots[i].X_est[:, k], k+1, False)
            # Due to difficulties with noncontinuous system ie. reflections off of walls,
            # Set velocities of est X to velocities of act X
            robots[i].X_est[2:, k+1] = robots[i].X_act[2:, k+1]
            
            # Propagate P; gives P_k+1 prior
            _, F, _, _, _, _, _ = robots[i].Dynamics(k)
            robots[i].P = (F @ robots[i].P) @ F.T + robots[i].G @ robots[i].w[1] @ robots[i].G.T
            
            ErrorBars(robots[i], k+1)
            F.fill(0)
        
        # 2a. Check if all actual X's is in sensors FoV
        # Currently prioritizes robot 2 to be targeted
        for i in range(num_sensors):
            if sensors[i].InFoV(robots[1].X_act[:, k+1]):
                sensors[i].SwitchTarget(robots[1])
            elif sensors[i].InFoV(robots[0].X_act[:, k+1]):
                sensors[i].SwitchTarget(robots[0])
            else:
                sensors[i].SwitchTarget(None)
        
        # 2b. Update X, P of sensors' targets
        for i in range(num_sensors):
            if sensors[i].target != None:
                # Take observation data from actual X
                sensors[i].target.Y_act[:, k+1], _ = sensors[i].Obs(sensors[i].target.X_act[:, k+1], True)
                # Take observation data from actual X
                sensors[i].target.Y_est[:, k+1], H = sensors[i].Obs(sensors[i].target.X_est[:, k+1], False)
                
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
                
    return robots, sensors