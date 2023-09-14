# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:33:21 2023

@author: tarun
"""

# Working Version
# Adding in KF, assumes measurements given as r_x and r_y of k+1 X

import sympy as sp
import numpy as np
from numpy import linalg as la

# Create parent class of robots and sensors ie. the modeling environment
class Entity(object):
    l = np.array([10, 10])                                                      # Length of room (x,y)
    dim_state = 4
    dim_msmt = 2
    
    time_start = 0
    time_end = 10
    h = .5
    t = np.arange(time_start, time_end + h, h)
    
    G = h*np.identity(dim_state)
    M = np.identity(dim_msmt)
    
    sig_bounds = 3

class Sensor(Entity):
    def __init__(self, sensor_position, field_of_view, R_sensor):
        self.pos = sensor_position
        self.FoV = field_of_view
        self.target = None
        self.v = (0, R_sensor)
        self.K = np.zeros((self.dim_state, self.dim_state))
    
    def InFoV(self, X):
        # Must always be called before Obs() is called
        
        # Returns True if both statements below are True
        in_x_range = self.FoV[0, 0] <= X[0] <= self.FoV[0, 1]
        in_y_range = self.FoV[1, 0] <= X[1] <= self.FoV[1, 1]
        return in_x_range and in_y_range 
    
    def SwitchTarget(self, new_target):
        self.target = new_target
        
    def Obs(self, X, is_act):
        # Will need to change when sensor position changes
        r_x, r_y, v_x, v_y = sp.symbols("r_x, r_y, v_x, v_y")
        X_k = np.array([[r_x], [r_y], [v_x], [v_y]])
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        obs = (H @ X_k).reshape(self.dim_msmt)
        
        for i in range(self.dim_msmt):
            obs[i] = obs[i].subs([(r_x, X[0]), (r_y, X[1]), (v_x, X[2]), (v_y, X[3])])
            
        if is_act == True:
            obs += self.M @ np.random.multivariate_normal(np.zeros(self.dim_msmt),self.v[1])
            
        return np.matrix(obs).astype(np.float64), H

class Robot(Entity):
    def __init__(self, vel, start_pos, Q_robot):
        self.X_act = np.zeros((self.dim_state, self.t.size))
        self.X_est = np.zeros((self.dim_state, self.t.size))
        self.X_act[:, 0] = np.array([start_pos[0], start_pos[1], vel[0], vel[1]]).reshape((self.dim_state,))
        self.X_est[:, 0] = np.array([start_pos[0], start_pos[1], vel[0], vel[1]]).reshape((self.dim_state,))
        self.w = (0, Q_robot / self.h)
        self.Y_act = np.zeros((self.dim_msmt, self.t.size))
        self.Y_act.fill(np.nan)
        self.Y_est = np.zeros((self.dim_msmt, self.t.size))
        self.Y_est.fill(np.nan)
        self.P = np.diag(np.array([1, 1, 0, 0]))
        self.error_bars = np.zeros((self.dim_state, self.t.size))
    
    def Dynamics(self, k_k):
        # Set up for 2 spatial dimensions ie. x, y
        r_x, r_y, v_x, v_y, k = sp.symbols("r_x, r_y, v_x, v_y, k")
        X_k = np.array([[r_x], [r_y], [v_x], [v_y]])
        
        F = np.array([[1, 0, self.h, 0,],
                      [0, 1, 0, self.h], 
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        X_k_prop = (F @ X_k).reshape(self.dim_state)
        
        return X_k_prop, np.array(F).astype(np.float64), r_x, r_y, v_x, v_y, k
        
    def Propagation(self, X_k, k_k, is_act):
        X_k_prop, _, r_x, r_y, v_x, v_y, k = self.Dynamics(k_k)
        
        for i in range(self.dim_state):
            X_k_prop[i] = X_k_prop[i].subs([(r_x, X_k[0]), (r_y, X_k[1]), (v_x, X_k[2]), (v_y, X_k[3]), (k, k_k)])
        
        if is_act == True:
            X_k_prop += self.G @ np.random.multivariate_normal(np.zeros(self.dim_state),self.w[1])

        # If propagation causes robot to leave boundary x = l[0], x = 0, y = l[0], y = 0 respectively
        # Reflects robot back into bounds and flips V as needed
        
        if X_k_prop[0] >= self.l[0]:
            X_k_prop[0] = 2*self.l[0] - X_k_prop[0]
            #v_x should always be (-) if hits right wall
            X_k_prop[2] = -1 * abs(X_k_prop[2])
            
        if X_k_prop[0] <= 0:
            X_k_prop[0] *= -1
            #v_x should always be (+) if hits left wall
            X_k_prop[2] = abs(X_k_prop[2])
            
        if X_k_prop[1] >= self.l[1]:
            X_k_prop[1] = 2*self.l[1] - X_k_prop[1]
            #v_x should always be (+) if hits top wall
            X_k_prop[3] = -1 * abs(X_k_prop[3])
            
        if X_k_prop[1] <= 0:
            X_k_prop[1] *= -1
            #v_x should always be (+) if hits bottom wall
            X_k_prop[3] = abs(X_k_prop[3])
        
        return np.matrix(X_k_prop).astype(np.float64)