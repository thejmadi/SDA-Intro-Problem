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

if __name__ == '__main__':
    num_runs = 4
    J_optimized = np.zeros(num_runs)
    
    # States in [r_x, r_y, v_x, v_y].T format
    
    # Call Kalman Filter
    J_optimized = fc.MultiMonteCarlo(num_runs)
    
    # Call plots
    #plot.OptimizedCost(J_optimized)
'''  
# Create cProfile Data
run = cProfile.Profile()
run.run("Main()")
run.dump_stats("cProfile.prof")

with open("cProfile.txt", "w") as txt:
    stats = pstats.Stats("cProfile.prof", stream = txt)
    stats.sort_stats("cumtime")
    stats.print_stats()
'''