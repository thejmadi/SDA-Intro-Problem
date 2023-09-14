# SDA-Intro-Problem
Space Domain Awareness Intro problem, deals with multi-target tracking, Kalman Filters, and optimization of sensor tasking.


Files Included:

Entities - Houses entity, sensor, and robot classes.

Functions - Houses KF and Errorbar functions.

Plots - Houses state estimation plots and enviroment visualization.

Main - Runs simulations and calls other files.


Currently:

Creates robots and sensor instances.

Calculates estimated and actual state and covariance of robots.

Sensors can be set to prioritize a certain robot.

Plots necessary graphs.


Up next:

Task sensor by minimizing a cost function. 


Assumptions:

2D

Problem is linear (speed of robots are held constant, mseasurements are x and y position).

Data Association Problem is neglected.

Birth and death problems associated with SDA is neglected.
