# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:33:21 2023

@author: tarun
"""

# Working Version
import Functions as fc
import Plots as plot
import sympy as sp
import numpy as np
from numpy import linalg as la
import multiprocessing as mlt
import itertools as it
import time

# Create Environment Class and 3 children classes
# Sensors, Robots, Optimization
# Only 1 Optimization instance is needed

class Environment(object):
    l = np.array([10, 10])                                                     # Length of room (x,y)
    dim_state, dim_msmt = 4, 2
    
    time_start, time_end = 0, 21
    timestep = 1.0
    t_array = np.arange(time_start, time_end + timestep, timestep)
    T = t_array.shape[0]
    
    MC_runs = 1
    
    G, M = timestep*np.identity(dim_state), np.identity(dim_msmt)
    
    sig_bounds = 3
    
    simple_rng = np.random.default_rng(24058)
    # Robot Parameters, Shown for 2 robots (x, y)
    #vel = np.array([[2, 2], [1, 2], [1, -2], [-1, -2], [-1, -1], [-1, 1], [-2, 2], [-1, 1]])
    #start_pos = np.array([[0, 0], [0, 5], [0, 10], [5, 10], [10, 10], [10, 5], [10, 0], [5, 0]])
    #vel = np.array([[0, 1], [0, 2]])
    #start_pos = np.array([[2.5, 0], [7, 0]])
    vel = simple_rng.uniform(-2, 2, size=(10, 2))
    start_pos = simple_rng.uniform(2, 8, size=(10, 2))
    Q_robot = (np.ones((10, 4)) @ np.diag([0.01, 0.01, 0, 0])) * np.abs(np.hstack((vel, start_pos)))
    N = 10
    
    # Sensor Parameters, Shown for 2 sensor, Only using first 1
    sensor_position = np.array([[0, 0], [0, 0]])
    field_of_view = np.array([[0, l[0]/2, 0, l[1]], [0, l[0]/2, 0, l[1]]])
    R_sensor = np.array([[0.005, 0.005], [0.005, 0.005]])
    S = 1
    
    # Policy index 0 is index 1 in t array (After 1 timestep)
    current_policy = np.ones((N, T - 1))/N
    all_policies = np.zeros((N, T-1, 4))
    all_policies[:, :, 0] = current_policy
    #current_policy = np.array([[0, 0, 0],
    #                           [0, 0, 0],
    #                           [1, 1, 1]])
    optimal_policy = np.zeros((current_policy.shape))
    min_cost = 100000
    min_iter = 0
    learn_rate = 0.05
    
    seed = 98765
        
class Sensor(Environment):
    def __init__(self, sensor_choice):
        self.Reset(sensor_choice)
    
    def Reset(self, sensor_choice):
        self.pos = self.sensor_position[sensor_choice]
        self.FoV = self.field_of_view[sensor_choice].reshape((self.dim_msmt, self.dim_msmt))
        self.target = None
        self.targets_over_time = np.zeros(self.T)
        self.v = (0, np.diag(self.R_sensor[sensor_choice]))
        self.K = np.zeros((self.dim_state, self.dim_state))
        self.robots_in_FoV = np.zeros((self.N, self.T-1))

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
        '''
        if is_act == True and is_multi == True:
            rand = rng.multivariate_normal(np.zeros(self.dim_msmt),self.v[1])
            obs += self.M @ rand
            
        # Random for single process
        elif is_act == True and is_multi == False:
            obs += self.M @ np.random.multivariate_normal(np.zeros(self.dim_msmt),self.v[1])
        '''
            
        return np.matrix(obs).astype(np.float64), H

class Robot(Environment):
    
    id_it = it.count()
    
    def __init__(self, robot_choice):
        self.Reset(robot_choice)
        self.id = next(self.id_it)
    
    def Reset(self, robot_choice):
        self.X_act = np.zeros((self.dim_state, self.T))
        self.X_est = np.zeros((self.dim_state, self.T))
        self.X_act[:, 0] = np.array([self.start_pos[robot_choice, 0], self.start_pos[robot_choice, 1], self.vel[robot_choice, 0], self.vel[robot_choice, 1]]).reshape((self.dim_state,))
        self.X_est[:, 0] = np.array([self.start_pos[robot_choice, 0], self.start_pos[robot_choice, 1], self.vel[robot_choice, 0], self.vel[robot_choice, 1]]).reshape((self.dim_state,))
        self.w = (0, np.diag(self.Q_robot[robot_choice]) / self.timestep)
        self.Y_act = np.zeros((self.dim_msmt, self.T))
        self.Y_act.fill(np.nan)
        self.Y_est = np.zeros((self.dim_msmt, self.T))
        self.Y_est.fill(np.nan)
        self.P = np.diag(np.array([1, 1, 0, 0]))
        self.error_bars = np.zeros((self.dim_state, self.T))
    
    def Dynamics(self, k_k):
        # Set up for 2 spatial dimensions ie. x, y
        r_x, r_y, v_x, v_y, k = sp.symbols("r_x, r_y, v_x, v_y, k")
        X_k = np.array([[r_x], [r_y], [v_x], [v_y]])
        
        F = np.array([[1, 0, self.timestep, 0,],
                      [0, 1, 0, self.timestep], 
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        X_k_prop = (F @ X_k).reshape(self.dim_state)
        
        return X_k_prop, np.array(F).astype(np.float64), r_x, r_y, v_x, v_y, k
        
    def Propagation(self, X_k, k_k, is_act, is_multi, rng):
        X_k_prop, _, r_x, r_y, v_x, v_y, k = self.Dynamics(k_k)
        
        for i in range(self.dim_state):
            X_k_prop[i] = X_k_prop[i].subs([(r_x, X_k[0]), (r_y, X_k[1]), (v_x, X_k[2]), (v_y, X_k[3]), (k, k_k)])
        
        '''
        if is_act == True and is_multi == True:
            rand = rng.multivariate_normal(np.zeros(self.dim_state),self.w[1])
            X_k_prop += self.G @ rand
        
        # Random for single process
        elif is_act == True and is_multi == False:
            X_k_prop += self.G @ np.random.multivariate_normal(np.zeros(self.dim_state),self.w[1])
        '''
        
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
    def __init__(self):
        self.Reset()
    
    def Reset(self):
        # Policy index 0 is index 1 in t array (After 1 timestep)
        self.frozen_policy = np.zeros((self.N, self.T-1))
        self.frozen_policy[:, :] = self.current_policy[:, :]
        # Matrix of NxT to be averaged across 1 set of MC runs
        self.J = np.zeros((self.N, self.T-1))
        # Matrix of NxT to be averaged across 1 set of frozen policy MC runs, overwritten for each frozen policy 
        self.frozen_J = np.zeros((self.N, self.T-1))
        # Matrix of NxT. Only 1st row & self.J & self.frozen_J are needed to fill 
        self.partial_J = np.zeros((self.N, self.T-1))
    
    # Updates either self.J or self.frozen_J 
    def UpdateJ(self, cov, n, t, is_frozen):
        if not is_frozen:
            #print("Robot #" + str(n) + " Timestep " + str(t))
            #print(cov)
            self.J[n, t] += np.trace(cov) / self.MC_runs
            #print(self.J)
        else:
            #print(is_frozen)
            self.frozen_J[n, t] += np.trace(cov) / self.MC_runs
        
    def UpdatePartialJ(self, n, t):
        self.partial_J[n, t] = np.sum(self.frozen_J)
        self.frozen_J.fill(0)

    def FreezePolicy(self, n, t):
        # Repair col t-1
        if t != 0 and n == 0:
            self.frozen_policy[:, t-1] = self.current_policy[:, t-1]
        self.frozen_policy[:, t].fill(0)   
        self.frozen_policy[n, t] = 1.0

    # TODO: Change all instances of policy use in Tasking to argument policy
    def Tasking(self, t, is_multi, is_frozen, rng, policy):
    # Tasks each sensor at every timestep (timestep t, bool is_perturbed, bool is_multi, obj rng)
        if is_multi == True:
            rand_num = rng.random()
        # Random for single Process
        elif is_multi == False:
            rand_num = np.random.random() 
            
        sensor_choice = 0
        prev = 0
        if not is_frozen: # If policy is unfrozen
            for n in range(0, self.N):
                if(prev <= rand_num < prev + policy[n, t]): # If rand_num inbetween interval described by adjacent policy elements (in col)
                    sensor_choice = n
                    break
                prev += policy[n, t]
        else: # If policy is frozen
            for n in range(0, self.N):
                if(prev <= rand_num < prev + self.frozen_policy[n, t]):
                    sensor_choice = n
                    break
                prev += self.frozen_policy[n, t]
        return sensor_choice # Outputs index of robot (in robots array) to keep track of 

class MCMC(Environment):
    num_chains = 1
    #num_burn = 15
    num_samples = 50000
    tol = 5e-6
    burn_window_len = 100
    #sample_window_len = 50
    
    def ParallelizeChains(self, robots, sensors, optimize, rng_class):
        self.Q_robot.fill(0.0)
        self.R_sensor.fill(0.0)
        final_policies_and_costs = []
        for m in range(self.num_chains):
            final_policies_and_costs.append(self.MetropolisHastings(robots, sensors, optimize, m+1, rng_class.rng_children[m]))
        # TODO: Link final policy to chain_id
        
        #with mlt.Pool(self.num_chains) as pool:
        #    start = time.time()
        #    multi_results = [pool.apply_async(self.MetropolisHastings, args=(robots, sensors, optimize, chain_id+1, rng_class.rng_children[chain_id])) for chain_id in range(1, self.num_chains+1)]
        #    for r in multi_results:
        #        final_policies_and_costs.append(r.get())
        #    print(time.time() - start)
        return final_policies_and_costs
    
    '''
    def ParallelizeMC(self, robots, sensors, optimize, rng_class, policy):
        expected_costs_per_robot = []
        #start = time.time()
        with mlt.Pool(self.num_processes) as pool:
            multi_results = [pool.apply_async(self.MonteCarlo, args=(robots, sensors, optimize, rng_class.rng_children[chain_id], policy)) for chain_id in range(self.MC_runs)]
            for r in multi_results:
                expected_costs_per_robot.append(r.get())
        #print("Time = " + str(time.time() - start))
        expected_costs_per_robot = np.asarray(expected_costs_per_robot)
        #print(expected_costs_per_robot)
        return np.sum(expected_costs_per_robot, axis=0)
    '''
    # TODO: Find better proposal technique
    def Proposal(self, current, last_change, rng_child):
        proposed = np.zeros((self.N, self.T-1))
        proposed[:, :] = current
        proposed_idx = (rng_child.integers(low=0, high=self.N), rng_child.integers(low=0, high=self.T-1))
        while proposed[proposed_idx] == 1:
            proposed_idx = (rng_child.integers(low=0, high=self.N), rng_child.integers(low=0, high=self.T-1))
        '''
        (proposed_row, proposed_col) = last_change
        while last_change[0] == proposed_row and last_change[1] == proposed_col:
            proposed_row = rng.integer(0, high=self.N)
            proposed_col = rng.integer(max(last_change[1], last_change[1]-1), high=min(last_change[1], last_change[1]+1))
        '''
        proposed[:, proposed_idx[1]].fill(0)
        proposed[proposed_idx] = 1
        return proposed, proposed_idx
        
    def Transition(self):
        pass
        
    def MonteCarlo(self, robots, sensors, optimize, rng_child, current_policy):
        expected_cost_per_robot = np.zeros(self.N)
        fc.KF(robots, sensors, optimize, True, False, rng_child, current_policy)
        fc.ResetInstances(robots, sensors)
        expected_cost_per_robot[:] = np.sum(optimize.J, axis=1)
        optimize.Reset()
        return expected_cost_per_robot
        
    # Find better way to construct initial policy
    def GenerateInitialPolicy(self, rng_child):
        ones_idx = rng_child.integers(self.N, size=self.T-1)
        policy = np.zeros((self.N, self.T-1))
        for t in range(self.T-1):
            policy[ones_idx[t], t] = 1
        #policy[1, :] = 1
        return policy
    
    def MetropolisHastings(self, robots, sensors, optimize, chain_id, rng_child, rng_class = None):
        print("Chain #" + str(chain_id))
        #costs = np.zeros((self.N, self.num_burn + self.num_samples))
        costs = []
        current_policy = np.zeros((self.N, self.T-1))
        current_policy[:, :] = self.GenerateInitialPolicy(rng_child)
        #samples = np.zeros((self.N, self.T-1, int(self.num_samples*0.1)+1))
        samples = np.zeros((self.N, self.T-1, self.num_samples))
        # Evaluate current policy
        curr_expected_cost_per_robot = np.zeros(self.N)
        curr_expected_cost_per_robot = self.MonteCarlo(robots, sensors, optimize, rng_child, current_policy)
        
        proposed_expected_cost_per_robot = np.zeros(self.N)
        change_idx = (rng_child.integers(low=0, high=self.N), rng_child.integers(low=0, high=self.T-1))
        costs.append(curr_expected_cost_per_robot.copy())
        burn_acceptance = 1
        burn_total = 1
        running_sum = np.sum(curr_expected_cost_per_robot)
        running_sum_recip = 1/np.sum(curr_expected_cost_per_robot)
        burn_running_avg = [running_sum / (burn_total)]
        burn_running_avg_recip = [running_sum_recip / (burn_total)]
        prev_avg = self.tol
        new_avg = burn_running_avg[-1]
        prev_avg_recip = self.tol
        new_avg_recip = burn_running_avg_recip[-1]
        start = time.time()
        best_policy = np.zeros((self.N, self.T-1))
        best_policy[:, :] = current_policy[:, :]
        best_policy_cost = np.zeros(self.N)
        best_policy_cost[:] = curr_expected_cost_per_robot[:]
        best_policy_idx = len(costs) -1
        while np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip >= self.tol:
            burn_total += 1
            proposed_policy, proposed_change_idx = self.Proposal(current_policy, change_idx, rng_child)
            proposed_expected_cost_per_robot[:] = self.MonteCarlo(robots, sensors, optimize, rng_child, proposed_policy)                
            
            # Acceptance Criteria
            cost_ratio = np.sum(proposed_expected_cost_per_robot) / np.sum(curr_expected_cost_per_robot)
            cost_ratio = 1 / cost_ratio
            threshold = rng_child.random()+0.85
            '''
            print("Proposed Change: " + str(proposed_change_idx))
            
            print()
            print("Current Cost: " + str(np.sum(curr_expected_cost_per_robot)))
            print("Proposed Cost: " + str(np.sum(proposed_expected_cost_per_robot)))
            print()
            
            print("Cost Ratio: " + str(cost_ratio))
            print("Threshold: " + str(threshold))
            print()
            '''
            #print(cost_ratio, threshold)
            if (cost_ratio >= 1) or (1 >= threshold):
                burn_acceptance += 1
                current_policy[:, :] = proposed_policy
                change_idx = proposed_change_idx
                curr_expected_cost_per_robot[:] = proposed_expected_cost_per_robot
            costs.append(curr_expected_cost_per_robot.copy()) # [J_1, J_2, ..., J_K]
            running_sum += np.sum(curr_expected_cost_per_robot)
            running_sum_recip += 1/np.sum(curr_expected_cost_per_robot)
            burn_running_avg.append(running_sum / (burn_total))
            burn_running_avg_recip.append(running_sum_recip / (burn_total))
            if burn_total % self.burn_window_len == 0:
                print("Burn Acceptance = " + str(burn_acceptance) + "/" + str(burn_total))
                print("Time = " + str(time.time() - start))
                start = time.time()
                prev_avg = new_avg
                new_avg = burn_running_avg[-1]
                prev_avg_recip = new_avg_recip
                new_avg_recip = burn_running_avg_recip[-1]
                print("Old = " + str(prev_avg))
                print("New = " + str(new_avg))
                print("Dif = " + str(np.abs(new_avg - prev_avg)))
                print("Rel = " + str(np.abs(new_avg - prev_avg)/prev_avg))                
                print("Old_Recip = " + str(prev_avg_recip))
                print("New_Recip = " + str(new_avg_recip))
                print("Dif_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)))
                print("Rel_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip))
                print()
                #print("New Avg = ")
                #print(str(new_avg))
                #print()
            if np.sum(curr_expected_cost_per_robot) <= np.sum(best_policy_cost):
                best_policy[:, :] = current_policy[:, :]
                best_policy_cost[:] = curr_expected_cost_per_robot[:]
                best_policy_idx = len(costs) -1
            #print(new_avg)
            #print(prev_avg)
            #else:
                #print("Rejected")
            #print()
            #if burn_total % 500 == 0:
                #self.MC_runs += 20
        samples[:, :, 0] = current_policy
        sample_totals = np.zeros(self.num_samples)
        sample_totals[0] = 1
        sample_acceptance = 0
        sample_idx = np.zeros(self.num_samples)
        converged_avg = burn_running_avg[-1]
        converged_avg_recip = burn_running_avg_recip[-1]
        running_sum = 0
        running_sum_recip = 0
        sample_running_avg = [converged_avg]
        sample_running_avg_recip = [converged_avg_recip]
        prev_avg = self.tol
        prev_avg_recip = self.tol
        new_avg = converged_avg
        new_avg_recip = converged_avg_recip
        plot.MCMCBurns(burn_total, np.asarray(burn_running_avg), "Burn AVG(J) vs Iteration", self.burn_window_len)
        plot.MCMCBurns(burn_total, np.asarray(burn_running_avg_recip), "Burn AVG(1/J) vs Iteration", self.burn_window_len)
        print("Burning complete...")
        print()
        self.num_samples -= burn_total
        for s in range(1, self.num_samples):
            #print("Sample #" + str(s))
            #while np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip >= self.tol:
            sample_totals[s] += 1
            proposed_policy, proposed_change_idx = self.Proposal(current_policy, change_idx, rng_child)
            proposed_expected_cost_per_robot[:] = self.MonteCarlo(robots, sensors, optimize, rng_child, proposed_policy)            
            # Acceptance Criteria
            cost_ratio = np.sum(proposed_expected_cost_per_robot) / np.sum(curr_expected_cost_per_robot)
            cost_ratio = 1 / cost_ratio
            #alpha = min(1, cost_ratio)
            
            threshold = rng_child.random()+0.85
            '''
            print("Proposed Change: " + str(proposed_change_idx))
            
            print()
            print("Current Cost: " + str(np.sum(curr_expected_cost_per_robot)))
            print("Proposed Cost: " + str(np.sum(proposed_expected_cost_per_robot)))
            print()
            
            print("Cost Ratio: " + str(cost_ratio))
            print("Threshold: " + str(threshold))
            print()
            '''
            if (cost_ratio >= 1) or (1 >= threshold):
                #print("Sample Accepted")
                current_policy[:, :] = proposed_policy
                change_idx = proposed_change_idx
                curr_expected_cost_per_robot[:] = proposed_expected_cost_per_robot
                sample_acceptance += 1
            #if sample_acceptance % self.avg_window_len == 0:
            costs.append(curr_expected_cost_per_robot.copy()) # [J_1, J_2, ..., J_K]
            '''
            running_sum += np.sum(curr_expected_cost_per_robot)
            running_sum_recip += 1/np.sum(curr_expected_cost_per_robot)
            sample_running_avg.append(running_sum / (sample_totals[s]))
            sample_running_avg_recip.append(running_sum_recip / (sample_totals[s]))
            if sample_totals[s] % self.sample_window_len == 0:
                print("Sample #" + str(s) + "/" + str(self.num_samples))
                print("Sample Iter" + str(sample_totals[s]))
                print("Time = " + str(time.time() - start))
                start = time.time()
                prev_avg = new_avg
                new_avg = sample_running_avg[-1]
                prev_avg_recip = new_avg_recip
                new_avg_recip = sample_running_avg_recip[-1]
                print("Old = " + str(prev_avg))
                print("New = " + str(new_avg))
                print("Dif = " + str(np.abs(new_avg - prev_avg)))
                print("Rel = " + str(np.abs(new_avg - prev_avg)/prev_avg))
                print("Old_Recip = " + str(prev_avg_recip))
                print("New_Recip = " + str(new_avg_recip))
                print("Dif_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)))
                print("Rel_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip))
                print()
            '''
            if np.sum(curr_expected_cost_per_robot) <= np.sum(best_policy_cost):
                best_policy[:, :] = current_policy[:, :]
                best_policy_cost[:] = curr_expected_cost_per_robot[:]
                best_policy_idx = len(costs) -1
            
            samples[:, :, s] = current_policy
            #sample_idx[s] = np.sum(sample_totals[:s])
            '''
            running_sum = 0
            running_sum_recip = 0
            new_avg = 0
            new_avg_recip = 0
            
            plot.PlotHeatMapAnimation(samples, samples.shape[2], "Samples", "Time t", "Robot n", "MCMC_trace_11_6_24_Det")
            plot.MCMCSamples(np.sum(sample_totals), np.asarray(sample_running_avg), "Sample AVG(J) vs Iteration", sample_totals)
            plot.MCMCSamples(np.sum(sample_totals), np.asarray(sample_running_avg_recip), "Sample AVG(1/J) vs Iteration", sample_totals)
            '''
            #else:
                #print("Rejected")
            #print()
        costs = np.asarray(costs).T
        print("Burn, Sample Acceptance Ratio = " + str(burn_acceptance/burn_total) + ", " + str(sample_acceptance/np.sum(sample_totals)))
        #print(sample_idx)
        #print(sample_idx.shape)
        #print(np.asarray(sample_running_avg))
        #print(np.asarray(sample_running_avg).shape)
        #plot.OptimizedCost(costs.shape[1]-1, costs, "Burn and Sample Cost vs Iteration")
        #plot.OptimizedCost(costs.shape[1]-1-burn_total, costs[burn_total:], "Sample Cost vs Iteration")
        return (costs, samples, burn_running_avg, burn_running_avg_recip, sample_running_avg, sample_running_avg_recip, sample_totals, best_policy, best_policy_cost, best_policy_idx)

class RNG(Environment):
    def __init__(self):
        # create the RNG that you want to pass around
        self.rng_parent = np.random.default_rng(self.seed)
        # get the SeedSequence of the passed RNG
        self.ss = self.rng_parent.bit_generator._seed_seq
        # create MC_runs initial independent states, last is for PerturbPolicy()
        self.child_states = self.ss.spawn(5*self.MC_runs+1)
        self.rng_children = [np.random.default_rng(self.child_states[k]) for k in range(5*self.MC_runs+1)]