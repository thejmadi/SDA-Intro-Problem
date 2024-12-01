# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:19:13 2024

@author: tarun
"""
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import multiprocessing as mlt
import time as time
'''
class SimulationVariables:
    def __init__(self, time_end, N):
        self.Reset(time_end, N)
        return
    
    def NewSim(self, time_end, N):
        self.time_start, self.time_end = 0, time_end
        self.timestep = 1
        self.t_array = np.arange(self.time_start, self.time_end + self.timestep, self.timestep)
        
        self.N, self.S, self.T = N, 1, self.t_array.shape[0]
        
        self.dim_state, self.dim_msmt = 4, 2
        
        self.G, self.M = self.timestep*np.identity(self.dim_state), np.identity(self.dim_msmt)
'''
'''
class SimulationVariables:
    time_start, time_end = 0, 10
    timestep = 1
    t_array = np.arange(time_start, time_end + timestep, timestep)
        
    N, S, T = 2, 1, t_array.shape[0]
        
    dim_state, dim_msmt = 4, 2
        
    G, M = timestep*np.identity(dim_state), np.identity(dim_msmt)
'''
class Simulation():
    is_stochastic = False
    cost_type = "Trace"
    def __init__(self, variables):
        self.time_start = variables["time_start"]
        self.time_end = variables["time_end"]
        self.time_step = variables["time_step"]
        self.N_total = variables["N_total"]
        self.S = variables["S"]
        self.dim_state, self.dim_msmt = variables["dim_state"], variables["dim_msmt"]
        self.t_array_total = np.arange(self.time_start, self.time_end + self.time_step, self.time_step)
        self.T_total = self.t_array_total.shape[0]
        
        self.robots = []
        self.sensors= []
        for n in range(self.N_total):
            self.robots.append(Robot(n, variables))
        for s in range(self.S):
            self.sensors.append(Sensor(s, variables))
        #self.ResetSimulation(N, T, partial_reset = False)
        return
    #timestep
    def ResetSimulation(self, N, T, partial_reset):
        self.N, self.T = N, T
        self.t_array = self.t_array_total[:T]
        
        if partial_reset:
            for n in range(self.N):
                self.robots[n].PartialReset()
        else:
            for n in range(self.N):
                self.robots[n].FullReset(self.N, self.T)
        for s in range(self.S):
            self.sensors[s].Reset(self.N, self.T)
            
        self.cost = np.zeros((self.N, self.T-1))
        return
    
    def Deterministic(self):
        for n in range(self.N_total):
            for t in range(1, self.T_total):
                self.robots[n].X_act_total[:, t], _ = self.robots[n].Propagate(t, self.robots[n].X_act_total[:, t-1], self.is_stochastic)
    
    def CalcCost(self, cov, n, t):
        if self.cost_type == "Trace":
            self.cost[n, t] = np.trace(cov)
        elif self.cost_type == "LogDet":
            pass
        return
    
    def SinglePass(self):
        for s in range(sim.S):
            sim.sensors[s].Tasking()
            sim.sensors[s].RobotInFoV(self.robots)
            sim.sensors[s].ObsHeatMap()
        sim.KalmanFilter()
    
    def RunIterative():
        pass
    
    def RunFullHorizon():
        pass
    
    def KalmanFilter(self, policy):
        I = np.identity(self.dim_state)
        G, M = self.time_step*np.identity(self.dim_state), np.identity(self.dim_msmt)
        
        for s in range(self.S):
            self.sensors[s].Tasking(policy)
            self.sensors[s].RobotInFoV(self.robots)
            self.sensors[s].ObsHeatMap()
        
        for t in range(self.T - 1):
            #print("k: ", k)
            # 1. Propagate
            for n in range(self.N):
                self.robots[n].ErrorBars(t)
                
                # Propagate est X; gives X_k+1 prior
                # Propagation using X_act????
                self.robots[n].X_est[:, t+1], F = self.robots[n].Propagate(t+1, self.robots[n].X_est[:, t], self.is_stochastic)
                # Due to difficulties with noncontinuous system ie. reflections off of walls,
                # Set velocities of est X to velocities of act X
                self.robots[n].X_est[2:, t+1] = self.robots[n].X_act[2:, t+1]
                
                # Propagate P; gives P_k+1 prior
                self.robots[n].P = (F @ self.robots[n].P) @ F.T + G @ self.robots[n].w[1] @ G
                
                #ErrorBars(robots[n], t+1)
                F.fill(0)
        
            # 2b. Update X, P of sensors' targets
            for s in range(self.S):
                target = self.robots[sim.sensors[s].chosen_robots[t]]
                if self.sensors[s].obs_heat_map[target.id, t]:
                    # Take observation data from actual X
                    target.Y_act[:, t+1], _ = self.sensors[s].Observation(target.X_act[:, t+1], self.is_stochastic)
                    # Take observation data from estimate X
                    target.Y_est[:, t+1], H = self.sensors[s].Observation(target.X_est[:, t+1], self.is_stochastic)
                    
                    # Remove bias from measurements
                    Y_error = (target.Y_act[:, t+1] - target.Y_est[:, t+1]).reshape((self.dim_msmt, 1))
    
                    # Calc K gain for each estimation
                    # Next line does not use M?????
                    self.sensors[s].K = target.P @ H.T @ la.pinv((H @ target.P) @ H.T + self.sensors[s].v[1])
    
                    # Update est X
                    target.X_est[:, t+1] = target.X_est[:, t+1] + (self.sensors[s].K @ (Y_error)).reshape(self.dim_state)
    
                    # Update est P
                    target.P = (I - self.sensors[s].K @ H) @ target.P
    
                    self.robots[n].ErrorBars(t+1)
                    H.fill(0)
                                
            # Compute J
            for n in range(self.N):
                self.CalcCost(self.robots[n].P, n, t)
        return self.cost
    
class Robot():
    l = np.array([10, 10])
    #N = 10
    #r_0_list = np.array([[0.5, 0], [1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [3.5, 0], [4, 0], [4.5, 0], [5, 0]])
    #v_0_list = np.array([[0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2]])
    simple_rng = np.random.default_rng(24058)
    r_0_list = simple_rng.uniform(2, 8, size=(10, 2))
    v_0_list = simple_rng.uniform(-2, 2, size=(10, 2))
    #Q_list = (np.ones((10, 4)) @ np.diag([0.01, 0.01, 0, 0])) * np.abs(np.hstack((v_0_list, r_0_list)))
    sigma_bounds = 3
    
    def __init__(self, id_num, variables):
        self.id = id_num
        self.time_start = variables["time_start"]
        self.time_end = variables["time_end"]
        self.time_step = variables["time_step"]
        self.N_total = variables["N_total"]
        self.dim_state, self.dim_msmt = variables["dim_state"], variables["dim_msmt"]
        self.t_array_total = np.arange(self.time_start, self.time_end + self.time_step, self.time_step)
        self.T_total = self.t_array_total.shape[0]
        self.X_act_total = np.zeros((self.dim_state, self.T_total))
        self.X_act_total[:, 0] = np.array([self.r_0_list[self.id, 0], self.r_0_list[self.id, 1], self.v_0_list[self.id, 0], self.v_0_list[self.id, 1]]).reshape((self.dim_state,))
        self.Q_list = (np.ones((self.N_total, 4)) @ np.diag([0.01, 0.01, 0, 0])) * np.abs(np.hstack((self.v_0_list, self.r_0_list)))
        self.w = (0, np.diag(self.Q_list[self.id]) / self.time_step)
        #self.FullReset()
    
    def FullReset(self, N, T):
        self.N, self.T = N, T
        self.t_array = self.t_array_total[:T]
        self.X_act = np.zeros((self.dim_state, self.T))
        self.X_act[:, :] = self.X_act_total[:self.dim_state, :self.T]
        self.X_est = np.zeros((self.dim_state, self.T))
        self.X_est[:, 0] = np.array([self.r_0_list[self.id, 0], self.r_0_list[self.id, 1], self.v_0_list[self.id, 0], self.v_0_list[self.id, 1]]).reshape((self.dim_state,))
        
        self.Y_act = np.zeros((self.dim_msmt, self.T))
        self.Y_est = np.zeros((self.dim_msmt, self.T))
        self.Y_act.fill(np.nan)
        self.Y_est.fill(np.nan)
        
        self.P = np.diag(np.array([1, 1, 0, 0]))
        self.error_bars = np.zeros((self.dim_state, self.T))
        
    def PartialReset(self):
        self.X_est = np.zeros((self.dim_state, self.T))
        self.X_est[:, 0] = np.array([self.r_0_list[self.id, 0], self.r_0_list[self.id, 1], self.v_0_list[self.id, 0], self.v_0_list[self.id, 1]]).reshape((self.dim_state,))
        
        self.Y_act = np.zeros((self.dim_msmt, self.T))
        self.Y_est = np.zeros((self.dim_msmt, self.T))
        self.Y_act.fill(np.nan)
        self.Y_est.fill(np.nan)
        
        self.P = np.diag(np.array([1, 1, 0, 0]))
        self.error_bars = np.zeros((self.dim_state, self.T))
    
    def Propagate(self, t, X_prev, is_stochastic):
        Phi = np.array([[1, 0, self.time_step, 0,],
                        [0, 1, 0, self.time_step], 
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        X_next = Phi @ X_prev.reshape((self.dim_state, 1))
        
        #if self.is_stochastic:
        #    X_next += self.G @ rng.multivariate_normal(np.zeros(self.dim_state),self.w[1])
        
        if X_next[0] >= self.l[0]:
            X_next[0] = 2*self.l[0] - X_next[0]
            #v_x should always be (-) if hits right wall
            X_next[2] = -1 * abs(X_next[2])
            
        if X_next[0] <= 0:
            X_next[0] *= -1
            #v_x should always be (+) if hits left wall
            X_next[2] = abs(X_next[2])
            
        if X_next[1] >= self.l[1]:
            X_next[1] = 2*self.l[1] - X_next[1]
            #v_x should always be (-) if hits top wall
            X_next[3] = -1 * abs(X_next[3])
            
        if X_next[1] <= 0:
            X_next[1] *= -1
            #v_x should always be (+) if hits bottom wall
            X_next[3] = abs(X_next[3])
        
        return X_next[:, 0], Phi
    
    def ErrorBars(self, k):
        self.error_bars[:, k] = self.sigma_bounds * np.sqrt(np.diag(self.P).astype(float))
        return

class Sensor():
    FoV_list = np.array([[0, 5, 0, 10]])
    R_list = np.array([[0.005, 0.005]])
    
    def __init__(self, id_num, variables):
        self.sensor_id = id_num
        self.sensor_choice = 0
        self.v = (0, np.diag(self.R_list[self.sensor_choice]))
        self.time_start = variables["time_start"]
        self.time_end = variables["time_end"]
        self.time_step = variables["time_step"]
        self.N_total = variables["N_total"]
        self.dim_state, self.dim_msmt = variables["dim_state"], variables["dim_msmt"]
        self.t_array_total = np.arange(self.time_start, self.time_end + self.time_step, self.time_step)
        self.T_total = self.t_array_total.shape[0]
        #self.current_policy = np.ones((self.N, self.T-1))/self.N
        
    def Reset(self, N, T):
        self.N, self.T = N, T
        self.t_array = self.t_array_total[:T]
        self.FoV = self.FoV_list[self.sensor_choice].reshape((self.dim_msmt, self.dim_msmt))
        self.FoV_heat_map = np.zeros((self.N, self.T-1))
        self.chosen_robots = np.zeros(self.T-1, dtype=int)
        self.targets = np.zeros((self.N, self.T-1))
        self.obs_heat_map = np.zeros((self.N, self.T-1))
    
    def RobotInFoV(self, robots):
        # Must always be called before Obs() is called
        # Returns True if both statements below are True
        for n in range(self.N):
            X = robots[n].X_act[0, 1:]
            Y = robots[n].X_act[1, 1:]
            in_x_range = np.logical_and(np.greater_equal(X, self.FoV[0, 0]), np.less_equal(X, self.FoV[0, 1]))
            in_y_range = np.logical_and(np.greater_equal(Y, self.FoV[1, 0]), np.less_equal(Y, self.FoV[1, 1]))
            self.FoV_heat_map[n, :] = np.logical_and(in_x_range, in_y_range)
        return
    
    def SwitchTarget(self, new_target, k, robot_id):
        self.target = new_target
        # TODO: Check +1 below
        self.targets_over_time[k] = robot_id+1
    
    def Observation(self, X, is_stochastic):
        # Will need to change when sensor position changes
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        Y = H @ X.reshape((self.dim_state, 1))
        
        #if is_stochastic:
        #    Y += self.M @ rng.multivariate_normal(np.zeros(self.dim_state),self.v[1])
            
        return Y.reshape(self.dim_msmt), H
    
    def Tasking(self, policy):
        rng = np.random.default_rng(35635)
        rand_nums = rng.random(self.T-1)
        for n in range(0, self.N):
            prev = np.sum(policy[:n, :], axis=0)
            curr = np.sum(policy[:n+1, :], axis=0)
            in_between = np.logical_and(np.greater_equal(rand_nums, prev), np.less_equal(rand_nums, curr))
            self.chosen_robots[in_between] = n
            self.targets[n, :] = in_between
            if np.sum(self.targets) == self.T:
                n = self.N+1
        return
    
    def ObsHeatMap(self):
        self.obs_heat_map[:, :] = np.logical_and(self.FoV_heat_map, self.targets)
        return

class RNG:
    def __init__(self, MC_num):
        # create the RNG that you want to pass around
        self.rng_parent = np.random.default_rng(43823)
        # get the SeedSequence of the passed RNG
        self.ss = self.rng_parent.bit_generator._seed_seq
        # create MC_runs initial independent states, last is for PerturbPolicy()
        self.child_states = self.ss.spawn(MC_num+1)
        self.rng_child = [np.random.default_rng(self.child_states[k]) for k in range(MC_num+1)]
        self.num_to_burn = np.zeros(MC_num+1, dtype=int)

class Optimization:
    def __init__(self):
        self.MC_num = 100
        self.num_processes = 1
        self.burn = np.zeros(self.MC_run, dtype=int)
    '''
    def Process(self, use_monte_carlo, rng_class, child_id):
        # 
        if sim.is_stochastic == False:
            for n in range(sim.N):
                for t in range(1, sim.T):
                    sim.robots[n].X_act[:, t], _ = sim.robots[n].Propagate(t, sim.is_stochastic)
        if use_monte_carlo:
            self.MonteCarlo(sim, rng_class)
        else:
            _, _ = self.SinglePass(sim, rng_class[])
        
        return 
    '''
    def MonteCarlo(self, sim, rng_class):
        multi_results = []
        with mlt.Pool(self.num_processes) as pool:
            temp_results = [pool.apply_async(self.SinglePass, args=(sim, rng_class.rng_child[child_id], self.burn[child_id], child_id)) for child_id in range(self.MC_num)]
            for r in temp_results:
                multi_results.append(r.get())
                self.burn[multi_results[-1][1]] += multi_results[-1][0]
        return
    
    def SinglePass(self, sim, rng, burn=None, mlt_id=None):
        if mlt_id != None:
            pass
        for s in range(sim.S):
            sim.sensors[s].Tasking()
            sim.sensors[s].RobotInFoV(sim.robots)
            sim.sensors[s].ObsHeatMap()
        sim.KalmanFilter(sim.robots, sim.sensors)
        return burn, mlt_id

    

class MCMC():
    num_processes = 6
    num_samples = 1000
    num_burn = 1000
    tol = 5e-4
    burn_window_len = 500
    sample_window_len = 500
    independent_samples = False
    
    def __init__(self, variables):
        self.time_start = variables["time_start"]
        self.time_end = variables["time_end"]
        self.time_step = variables["time_step"]
        self.N_total = variables["N_total"]
        self.t_array_total = np.arange(self.time_start, self.time_end + self.time_step, self.time_step)
        self.T_total = self.t_array_total.shape[0]
        # TODO: Move to Simulation class
        return
    
    def Reset(self, N, T):
        self.N, self.T = N, T
        self.t_array = self.t_array_total[:T]
        return
    
    def ParallelizeChains(self, sim, MC_num, rng_class):
        final_policies_and_costs = []
        #for m in range(1):
        #    final_policies_and_costs.append(self.MetropolisHastings(sim, m+1, rng_class.rng_child[0]))
        with mlt.Pool(self.num_processes) as pool:
        #    start = time.time()
            multi_results = [pool.apply_async(self.MetropolisHastings, args=(sim, m, rng_class.rng_child[m], rng_class.num_to_burn[m])) for m in range(1, MC_num+1)]
            for r in multi_results:
                final_policies_and_costs.append(r.get())
            for r in final_policies_and_costs:
                rng_class.num_to_burn[r[8]] += r[9]
        #    print(time.time() - start)
        return final_policies_and_costs
    
    def Proposal(self, current_policy, rng_child):
        proposed_policy = np.zeros((self.N, self.T-1))
        proposed_policy[:, :] = current_policy
        proposed_idx = (rng_child.integers(low=0, high=self.N), rng_child.integers(low=0, high=self.T-1))
        rng_burn = 2
        while proposed_policy[proposed_idx] == 1:
            proposed_idx = (rng_child.integers(low=0, high=self.N), rng_child.integers(low=0, high=self.T-1))
            rng_burn += 2
        proposed_policy[:, proposed_idx[1]].fill(0)
        proposed_policy[proposed_idx] = 1
        return proposed_policy, rng_burn
    
    def GenerateInitialPolicy(self, rng_child):
        ones_idx = rng_child.integers(self.N, size=self.T-1)
        rng_burn = self.T-1
        policy = np.zeros((self.N, self.T-1))
        policy[ones_idx, range(self.T-1)] = 1
        return policy, rng_burn
    
    def MetropolisHastings(self, sim, chain_id, rng_child, num_to_burn):
        rng_child.integers(self.N, size=num_to_burn)
        #print("Chain #" + str(chain_id))
        costs = []
        
        current_policy = np.zeros((self.N, self.T-1))
        current_policy[:, :], temp_burn = self.GenerateInitialPolicy(rng_child)
        rng_burn = temp_burn
        # Evaluate costs
        current_costs = np.zeros(self.N)
        current_costs[:] = np.sum(sim.KalmanFilter(current_policy), axis=1)            
        sim.ResetSimulation(self.N, self.T, partial_reset=True)
        proposed_costs = np.zeros(self.N)
        costs.append(current_costs.copy())
        
        # Counters
        burn_acceptance = 1
        burn_total = 1
        
        # Avg information
        running_sum_recip = np.sum(current_costs)
        burn_running_avg_recip = [running_sum_recip / (burn_total)]
        prev_avg_recip = burn_running_avg_recip[-1] + 3*self.tol
        new_avg_recip = burn_running_avg_recip[-1]
        
        # Best policy information
        best_policy = np.zeros((self.N, self.T-1))
        best_policy[:, :] = current_policy[:, :]
        best_policy_cost = np.zeros(self.N)
        best_policy_cost[:] = current_costs[:]
        best_policy_idx = len(costs) -1
        
        start = time.time()
        #while np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip >= self.tol:
        for i in range(1, self.num_burn):
            burn_total += 1
            proposed_policy, temp_burn = self.Proposal(current_policy, rng_child)
            rng_burn += temp_burn
            proposed_costs[:] = np.sum(sim.KalmanFilter(proposed_policy), axis=1)          
            sim.ResetSimulation(self.N, self.T, partial_reset=True)              
            
            # Acceptance Criteria
            cost_ratio = np.sum(current_costs) / np.sum(proposed_costs)
            threshold = rng_child.random()+0.85
            rng_burn += 1
            
            #print(cost_ratio, threshold)
            if (cost_ratio >= 1) or (1 >= threshold):
                burn_acceptance += 1
                current_policy[:, :] = proposed_policy
                current_costs[:] = proposed_costs
            
            costs.append(current_costs.copy()) # [J_1, J_2, ..., J_K]
            running_sum_recip += np.sum(current_costs)
            burn_running_avg_recip.append(running_sum_recip / (burn_total))
            
            if burn_total % self.burn_window_len == 0:
                #print("Burn Acceptance = " + str(burn_acceptance) + "/" + str(burn_total))
                #print("Time = " + str(time.time() - start))
                start = time.time()
                prev_avg_recip = new_avg_recip
                new_avg_recip = burn_running_avg_recip[-1]
                #print("Old_Recip = " + str(prev_avg_recip))
                #print("New_Recip = " + str(new_avg_recip))
                #print("Dif_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)))
                #print("Rel_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip))
                #print()
            
            # Best policy update
            if np.sum(current_costs) <= np.sum(best_policy_cost):
                best_policy[:, :] = current_policy[:, :]
                best_policy_cost[:] = current_costs[:]
                best_policy_idx = len(costs) -1
        
        samples = np.zeros((self.N, self.T-1, self.num_samples))
        samples[:, :, 0] = current_policy
        sample_totals = np.zeros(self.num_samples)
        sample_totals[0] = 1
        sample_acceptance = 0
        sample_idx = np.zeros(self.num_samples)
        
        converged_avg_recip = burn_running_avg_recip[-1]
        running_sum_recip = 0
        sample_running_avg_recip = [converged_avg_recip / sample_totals[0]]
        prev_avg_recip = converged_avg_recip + 3*self.tol
        new_avg_recip = converged_avg_recip
        #plot.MCMCBurns(burn_total, np.asarray(burn_running_avg), "Burn AVG(J) vs Iteration", self.burn_window_len)
        #plot.MCMCBurns(burn_total, np.asarray(burn_running_avg_recip), "Burn AVG(1/J) vs Iteration", self.burn_window_len)
        #print("Burning complete...")
        #print()
        for s in range(1, self.num_samples):
            #while np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip >= self.tol:
            sample_totals[s] += 1
            proposed_policy, temp_burn = self.Proposal(current_policy, rng_child)
            rng_burn += temp_burn
            proposed_costs[:] = np.sum(sim.KalmanFilter(proposed_policy), axis=1)          
            sim.ResetSimulation(self.N, self.T, partial_reset=True)
            
            # Acceptance Criteria
            cost_ratio = np.sum(current_costs) / np.sum(proposed_costs)
            threshold = rng_child.random()+0.85
            rng_burn += 1
            
            if (cost_ratio >= 1) or (1 >= threshold):
                current_policy[:, :] = proposed_policy
                current_costs[:] = proposed_costs
                sample_acceptance += 1
            costs.append(current_costs.copy()) # [J_1, J_2, ..., J_K]
            '''
            running_sum_recip += np.sum(current_costs)
            sample_running_avg_recip.append(running_sum_recip / (sample_totals[s]))
            if sample_totals[s] % self.sample_window_len == 0:
                print("Sample #" + str(s) + "/" + str(self.num_samples))
                print("Sample Iter" + str(sample_totals[s]))
                print("Time = " + str(time.time() - start))
                start = time.time()
                prev_avg_recip = new_avg_recip
                new_avg_recip = sample_running_avg_recip[-1]
                print("Old_Recip = " + str(prev_avg_recip))
                print("New_Recip = " + str(new_avg_recip))
                print("Dif_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)))
                print("Rel_Recip = " + str(np.abs(new_avg_recip - prev_avg_recip)/prev_avg_recip))
                print()
            '''
            if np.sum(current_costs) <= np.sum(best_policy_cost):
                best_policy[:, :] = current_policy[:, :]
                best_policy_cost[:] = current_costs[:]
                best_policy_idx = len(costs) -1
            
            samples[:, :, s] = current_policy
            #sample_idx[s] = np.sum(sample_totals[:s])
                
            #running_sum_recip = 0
            #prev_avg_recip = converged_avg_recip + 3*self.tol
            #new_avg_recip = converged_avg_recip
                
                #plot.PlotHeatMapAnimation(samples, samples.shape[2], "Samples", "Time t", "Robot n", "MCMC_trace_11_6_24_Det")
                #plot.MCMCSamples(np.sum(sample_totals), np.asarray(sample_running_avg), "Sample AVG(J) vs Iteration", sample_totals)
                #plot.MCMCSamples(np.sum(sample_totals), np.asarray(sample_running_avg_recip), "Sample AVG(1/J) vs Iteration", sample_totals)
                
        costs = np.asarray(costs).T
        #print("Burn, Sample Acceptance Ratio = " + str(burn_acceptance/burn_total) + ", " + str(sample_acceptance/np.sum(sample_totals)))
        #plot.OptimizedCost(costs.shape[1]-1, costs, "Burn and Sample Cost vs Iteration")
        #plot.OptimizedCost(costs.shape[1]-1-burn_total, costs[burn_total:], "Sample Cost vs Iteration")
        return (costs, samples, np.asarray(burn_running_avg_recip), np.asarray(sample_running_avg_recip), sample_totals, best_policy, best_policy_cost, best_policy_idx, chain_id, rng_burn)
    
class MCTS(Optimization):
    pass

class GradientDescent(Optimization):
    def __init__():
        pass
    
    def UpdatePolicy():
        pass

class Plotting():
    colors = ["blue", "green", "yellow"]
    def PlotRoom(self, robots, plot_est):
        plt.axvline(0, c='black', zorder=0)
        plt.axvline(robots[0].l[0], c='black', zorder=0)
        plt.axvspan(0, 5, ymin=1/12, ymax=11/12, alpha=0.3, color='gray', label='Sensor FoV')
        plt.axhline(0, c='black', zorder=0)
        plt.axhline(robots[0].l[1], c='black', zorder=0)
        for n in range(len(robots)):
            plt.plot(robots[n].X_act[0, :], robots[n].X_act[1, :])
            plt.scatter(robots[n].X_act[0, :], robots[n].X_act[1, :], c = self.colors[n%3], s = 40, label='Robot %i act' % (n+1))
            if plot_est:
                plt.scatter(robots[n].X_est[0, :], robots[n].X_est[1, :], c = 'r', s = 6, label = "Robot est")
        #plt.legend(loc="best")
        plt.title("Room Plot")
        plt.xlim(0 - 1, robots[0].l[0] + 1)
        plt.ylim(0 - 1, robots[0].l[1] + 1)
        plt.show()
        return

    def PlotHeatMap(self, data, title, x_label, y_label, normalize):
        im = plt.imshow(data, cmap="gray")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.colorbar(im)
        #if normalize:
        #    im.set_clim(0, 1)
        plt.show()
        return

    def PlotKF(self, robots):
        plots = ["x Position", "y Position", "x Velocity", "y Velocity"]
        
        # Plot robots' Est and Act States vs. Time
        for k in range(len(robots)):
            fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
            count = 0
            for i in range(graphs.shape[0]):
                for j in range(graphs.shape[1]):    
                    graphs[i][j].plot(robots[k].t_array, robots[k].X_act[count, :], c = self.colors[0], zorder=0, label = 'Act')
                    graphs[i][j].scatter(robots[k].t_array, robots[k].X_est[count, :], c = self.colors[1], linestyle='--', label='Est')
                    graphs[i][j].set_title(plots[count])
                    count += 1
            fig.suptitle("Robot %i States vs. Time" % (k+1))
            fig.legend(loc="upper right")
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Time")
            plt.show()
        
        # Plot est Error vs Time with error bars
        outside = 0
        total = 0
        for k in range(len(robots)):
            zero = np.zeros(robots[0].T)
            fig, graphs = plt.subplots(2, 2, sharex=True, figsize=(15,9))
            count = 0
            for i in range(graphs.shape[0]):
                for j in range(graphs.shape[1]):    
                    graphs[i][j].errorbar(robots[k].t_array, zero, yerr=robots[k].error_bars[count, :], fmt=' ', zorder=0)
                    graphs[i][j].scatter(robots[k].t_array, robots[k].X_act[count, :] - robots[k].X_est[count, :], c = 'r', s =10)
                    graphs[i][j].set_title(plots[count])
                    if count <= 1:
                        for m in range(robots[0].T-1):
                            total += 1
                            if abs(robots[k].X_act[count, m] - robots[k].X_est[count, m]) >= robots[k].error_bars[count, m]:
                                outside += 1
                    count += 1
            fig.suptitle("Robot %i Error with %i Sigma Error Bars vs. Time" % ((k+1), robots[k].sigma_bounds))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Time")
            plt.show()
            
        print(outside, "/", total, "estimations are outside of the errorbars (Position est only)")
        return

    def PlotHist(self, data, title, x_label, y_label):
        plt.hist(data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def PlotLine(self, x_data, data, title, x_label, y_label, plot_errors, errors=None, fmt=None):
        if plot_errors:
            plt.errorbar(x_data, data, yerr=errors, linestyle='None', fmt=fmt, capsize=3)
        else:
            plt.plot(x_data, data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def PlotHeatAnim():
        pass

if __name__ == '__main__':
    num_sims = 10
    sim_MC = 12
    var_incr = 20
    global_vars = {"N_total" : 10,
                   "S" : 1,
                   "time_start" : 0,
                   "time_end" : num_sims*var_incr+1,
                   "time_step" : 1,
                   "dim_state" : 4,
                   "dim_msmt" : 2}
    plotter = Plotting()
    rng_class = RNG(sim_MC)
    mcmc = MCMC(global_vars)
    sim = Simulation(global_vars)
    sim.ResetSimulation(global_vars["N_total"], var_incr+1, partial_reset=False)
    sim.Deterministic()
    mcmc_results = []
    costs = []
    samples = []
    burn_num = np.zeros((num_sims, sim_MC))
    burn_avg = np.zeros((num_sims, sim_MC))
    timer = []
    
    for i in range(1, num_sims+1):
        print("Sim # " + str(i))
        start = time.time()
        mcmc_results.append([])
        samples.append([])
        N, T = global_vars["N_total"], i*var_incr+1
        sim.ResetSimulation(N, T, partial_reset=False)
        #for k in range(sim_MC):
        sim.ResetSimulation(N, T, partial_reset=True)
        mcmc.Reset(N, T)
        mcmc_results[i-1].append(mcmc.ParallelizeChains(sim, sim_MC, rng_class))
        costs.append(np.sum(mcmc_results[i-1][-1][0][0], axis=0))
        #burn_num[i-1, k] = len(mcmc_results[i-1][-1][0][2])
        #burn_avg[i-1, k] = mcmc_results[i-1][-1][0][2][-1]
        samples[i-1].append(mcmc_results[i-1][-1][0][1])
            #plotter.PlotLine(mcmc_results[i-1][-1][0][2], "Burn Avg(1/J)", "Iteration", "Running Avg")
            #plotter.PlotRoom(sim.robots, plot_est=False)
        #plotter.PlotRoom(sim.robots, plot_est=False)
        #plotter.PlotLine(mcmc_results[i-1][0][3], "Sample Avg(1/J)", "Iteration", "Running Avg")
        #plotter.PlotHeatMap(np.sum(samples, axis=2)/samples.shape[2], "Aggregated Samples", "Timestep", "Robot", normalize=True)
        timer.append(time.time() - start)
    # Plots for varying T and N
    #burn_num_std = np.std(burn_num, axis=1)
    #plotter.PlotLine(range(var_incr, (num_sims+1)*var_incr, var_incr), np.sum(burn_num, axis=1)/sim_MC, "Number of Iterations Until Convergence at " + str(global_vars["N_total"]), "Number of Timesteps", "Avg Number of Burns", True, burn_num_std, 'o')
    #burn_avg_std = np.std(burn_avg, axis=1)
    #plotter.PlotLine(range(var_incr, (num_sims+1)*var_incr, var_incr), np.sum(burn_avg, axis=1)/sim_MC, "Converged Running Avg at " + str(global_vars["N_total"]), "Number of Timesteps", "Running Avg", True, burn_avg_std)
    
    #plotter.PlotLine(costs, "Burn and Sample Cost vs Iteration", "Iteration", "Cost")
    #plotter.PlotLine(costs[burn_num:], "Sample Cost vs Iteration", "Iteration", "Cost")
    #min_cost = np.min(costs)
    #min_idx = np.argmin(costs)
    #samples_totals = mcmc_results[0][4]
    #samples_totals[0] = burn_num

    ''' Diagnostic Plots
    plotter.PlotRoom(sim.robots, plot_est=False)
    plotter.PlotHeatMap(sim.sensors[0].targets, "Tasking Heat Map", "Timestep", "Robot", normalize=True)
    plotter.PlotHeatMap(sim.sensors[0].FoV_heat_map, "FoV Heat Map", "Timestep", "Robot", normalize=True)
    plotter.PlotHeatMap(sim.sensors[0].obs_heat_map, "Obs Heat Map", "Timestep", "Robot", normalize=True)
    plotter.PlotHeatMap(sim.cost, "Cost Heat Map", "Timestep", "Robot", normalize=True)
    plotter.PlotKF(sim.robots)
    '''


