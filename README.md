# SDA-Intro-Problem
Space Domain Awareness Intro problem, deals with multi-target tracking, Kalman Filters, and optimization of sensor tasking.


Files Included:

Entities - Houses environment, sensor, robot, and optimize classes.

Functions - Houses KF, Errorbar, ResetInstances, and MonteCarlo functions.

Plots - Houses state estimation plots and enviroment visualization.

Main - Runs simulations and calls other files. Outputs cProfile data to text file.


Currently:

Creates robots, sensor, and optimize instances.

Calculates estimated and actual state and covariance of robots.

Sensors tasked based on uniform probability choice for each robot in field of view.

Plots necessary graphs.

Runs a batch of Monte Carlo simulations.


Up next:

Add in multiprocessing for Monte Carlo simulations. 

Optimize tasking method using a gradient descent method over batches of Monte Carlo sims.


Assumptions:

2D

Problem is linear (speed of robots are held constant, mseasurements are x and y position).

Data Association Problem is neglected.

Birth and death problems associated with SDA is neglected.
