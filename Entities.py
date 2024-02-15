# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:33:21 2023

@author: tarun
"""

# Working Version

import sympy as sp
import numpy as np
from numpy import linalg as la
import itertools as it

# Create Environment Class and 3 children classes
# Sensors, Robots, Optimization
# Only 1 Optimization instance is needed

class Environment(object):
    l = np.array([10, 10])                                                      # Length of room (x,y)
    dim_state = 4
    dim_msmt = 2
    
    time_start = 0
    time_end = 4
    h = .4
    t = np.arange(time_start, time_end + h, h)
    
    MC_runs = 200
    output_file_name = "out.txt"
    
    G = h*np.identity(dim_state)
    M = np.identity(dim_msmt)
    
    sig_bounds = 3
    
    # Robot Parameters, Shown for 2 robots
    vel = np.array([[2, 1], [1, 2]])
    start_pos = np.array([[5, 0], [0, 5]])
    Q_robot = np.array([[0.25, 0.25, 0, 0], [0.25, 0.25, 0, 0]])
    number_robots = 2;
    
    # Sensor Parameters, Shown for 2 sensor, Only using first 1
    sensor_position = np.array([[0, 0], [0, 0]])
    field_of_view = np.array([[0, l[0]/2, 0, l[1]], [0, l[0]/2, 0, l[1]]])
    R_sensor = np.array([[0.25, 0.25], [0.25, 0.25]])
    number_sensors = 1
    
    # Policy index 0 is index 1 in t array (After 1 timestep)
    current_policy = np.ones((number_robots, t.shape[0] - 1))/number_robots
    #current_policy = np.array([[0, 1, 0],
    #                          [1, 0, 1]])
    optimal_policy = np.zeros((current_policy.shape))
    min_cost = 100000
    min_iter = 0
    perturb_const_up = 0.1
    perturb_const_low = 0.09
    learn_rate = 0.004
    
    seed = 98765
        
class Sensor(Environment):
    def __init__(self, sensor_choice):
        self.Reset(sensor_choice)
    
    def Reset(self, sensor_choice):
        self.pos = self.sensor_position[sensor_choice]
        self.FoV = self.field_of_view[sensor_choice].reshape((self.dim_msmt, self.dim_msmt))
        self.target = None
        self.targets_over_time = np.zeros(self.t.size)
        self.v = (0, np.diag(self.R_sensor[sensor_choice]))
        self.K = np.zeros((self.dim_state, self.dim_state))
        self.robots_in_FoV = []

    def InFoV(self, X):
        # Must always be called before Obs() is called
        
        # Returns True if both statements below are True
        in_x_range = self.FoV[0, 0] <= X[0] <= self.FoV[0, 1]
        in_y_range = self.FoV[1, 0] <= X[1] <= self.FoV[1, 1]
        return in_x_range and in_y_range 
    
    def SwitchTarget(self, new_target, k, robot_id):
        self.target = new_target
        self.targets_over_time[k] = robot_id+1
        
    def Obs(self, X, is_act, is_multi, rng):
        # Will need to change when sensor position changes
        r_x, r_y, v_x, v_y = sp.symbols("r_x, r_y, v_x, v_y")
        X_k = np.array([[r_x], [r_y], [v_x], [v_y]])
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        obs = (H @ X_k).reshape(self.dim_msmt)
        
        for i in range(self.dim_msmt):
            obs[i] = obs[i].subs([(r_x, X[0]), (r_y, X[1]), (v_x, X[2]), (v_y, X[3])])
            
        if is_act == True and is_multi == True:
            rand = rng.multivariate_normal(np.zeros(self.dim_msmt),self.v[1])
            obs += self.M @ rand
            
        # Random for single process
        elif is_act == True and is_multi == False:
            obs += self.M @ np.random.multivariate_normal(np.zeros(self.dim_msmt),self.v[1])

            
        return np.matrix(obs).astype(np.float64), H

class Robot(Environment):
    
    id_it = it.count()
    
    def __init__(self, robot_choice):
        self.Reset(robot_choice)
        self.id = next(self.id_it)
    
    def Reset(self, robot_choice):
        self.X_act = np.zeros((self.dim_state, self.t.size))
        self.X_est = np.zeros((self.dim_state, self.t.size))
        self.X_act[:, 0] = np.array([self.start_pos[robot_choice, 0], self.start_pos[robot_choice, 1], self.vel[robot_choice, 0], self.vel[robot_choice, 1]]).reshape((self.dim_state,))
        self.X_est[:, 0] = np.array([self.start_pos[robot_choice, 0], self.start_pos[robot_choice, 1], self.vel[robot_choice, 0], self.vel[robot_choice, 1]]).reshape((self.dim_state,))
        self.w = (0, np.diag(self.Q_robot[robot_choice]) / self.h)
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
        
    def Propagation(self, X_k, k_k, is_act, is_multi, rng):
        X_k_prop, _, r_x, r_y, v_x, v_y, k = self.Dynamics(k_k)
        
        for i in range(self.dim_state):
            X_k_prop[i] = X_k_prop[i].subs([(r_x, X_k[0]), (r_y, X_k[1]), (v_x, X_k[2]), (v_y, X_k[3]), (k, k_k)])
        
        if is_act == True and is_multi == True:
            rand = rng.multivariate_normal(np.zeros(self.dim_state),self.w[1])
            X_k_prop += self.G @ rand
        
        # Random for single process
        elif is_act == True and is_multi == False:
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
            #v_x should always be (-) if hits top wall
            X_k_prop[3] = -1 * abs(X_k_prop[3])
            
        if X_k_prop[1] <= 0:
            X_k_prop[1] *= -1
            #v_x should always be (+) if hits bottom wall
            X_k_prop[3] = abs(X_k_prop[3])
        
        return np.matrix(X_k_prop).astype(np.float64)

class Optimization(Environment):
    def __init__(self, num_robots):
        self.J_runs_t_n = np.zeros(self.number_robots) # Cost per robot per timestep
        self.J_runs_t = np.zeros(self.t.size) # Cost for all robots per timestep
        self.J_runs = np.zeros(self.MC_runs) # Cost for all robots for all timesteps
        self.multi_J_runs = 0 # Cost for all robots for all timesteps (Multiprocessing)
        self.J = 0 # Cost for all robots for all timesteps for all runs
        self.perturbed_policy = np.zeros(self.current_policy.shape)
        
    def Reset1(self):
        self.J_runs_t_n.fill(0)
        self.J_runs_t.fill(0)
    
    def Reset2(self):
        self.J_runs.fill(0)
        self.J = 0
    
    def Reset3(self):
        self.perturbed_policy.fill(0)
    
    def CostPerRobot(self, cov, i):
        # Should be Tr(cov), does not include velocity parts currently
        self.J_runs_t_n[i] = cov[0, 0] + cov[1, 1]
    
    def CostPerTimestep(self, k):
        self.J_runs_t[k] = np.sum(self.J_runs_t_n)
    
    def CostPerRun(self, n):
        self.J_runs[n] = np.sum(self.J_runs_t)
    
    def MultiCostPerRun(self):
        self.multi_J_runs = np.sum(self.J_runs_t)
    
    def CostTotal(self):
        self.J = np.sum(self.J_runs) / self.MC_runs
    

    def Tasking(self, t, is_perturbed, is_multi, rng):
    # Tasks each sensor at every timestep (timestep t, bool is_perturbed, bool is_multi, obj rng)
        if is_multi == True:
            rand_num = rng.random()
        # Random for single Process
        elif is_multi == False:
            rand_num = np.random.random()
            
            
        sensor_choice = 0
        prev = 0
        if not is_perturbed: # If policy is unperturbed
            for curr in range(0, self.number_robots):
                if(prev <= rand_num < prev + self.current_policy[curr, t]): # If rand_num inbetween interval described by adjacent policy elements (in col)
                    sensor_choice = curr
                    break
                prev += self.current_policy[curr, t]
        else: # If policy is perturbed
            for curr in range(0, self.number_robots):
                if(prev <= rand_num < prev + self.perturbed_policy[curr, t]):
                    sensor_choice = curr
                    break
                prev += self.perturbed_policy[curr, t]
        return sensor_choice # Outputs index of robot (in robots array) to keep track of 
    
    def PerturbPolicy(self, is_multi = False, rng = None):
        min_val = np.min(self.current_policy, axis=0)
        for col in range(0, self.current_policy.shape[1]):
            perturb_bounds_low = self.perturb_const_low * min_val[col]
            perturb_bounds_up = self.perturb_const_up * min_val[col]
            
            if is_multi == True:
                pos_neg = rng.choice([-1, 1])
                perturb_mag = rng.uniform(perturb_bounds_low, perturb_bounds_up)
            
            # Random for single Process
            elif is_multi == False:
                pos_neg = np.random.choice([-1, 1])
                perturb_mag = np.random.uniform(perturb_bounds_low, perturb_bounds_up)
            
            self.perturbed_policy[:1, col] = pos_neg * perturb_mag
            pos_neg *= -1
            self.perturbed_policy[1:, col] = pos_neg * perturb_mag
        self.perturbed_policy += self.current_policy
    
class RNG(Environment):
    def __init__(self):
        # create the RNG that you want to pass around
        self.rng_parent = np.random.default_rng(self.seed)
        # get the SeedSequence of the passed RNG
        self.ss = self.rng_parent.bit_generator._seed_seq
        # create MC_runs initial independent states, last is for PerturbPolicy()
        self.child_states = self.ss.spawn(self.MC_runs+1)
        self.rng_children = [np.random.default_rng(self.child_states[k]) for k in range(self.MC_runs+1)]