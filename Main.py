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
import time as time

if __name__ == '__main__':
    num_runs = 20
    J_optimized = np.zeros(num_runs)
    
    # States in [r_x, r_y, v_x, v_y].T format
    start = time.time()
    # Call Kalman Filter
    J_optimized = fc.MultiMonteCarlo(num_runs)
    end = time.time()
    print("Total time = ", end - start)
    # Call plots
    plot.OptimizedCost(num_runs, J_optimized)