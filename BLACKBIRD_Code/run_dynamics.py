# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:16:00 2022

This is the file to run which plots the dynamics using runge Kutta 4 integration method
"""

import matplotlib.pyplot as plt
import aero_params_benji as MAV
import mav_dynamics_benji as dynamics


#time step for integraion
dt = .1

#initialize the dynamics class
test = dynamics.MavDynamics(dt)

#make array for with initial conditions 
x = [MAV.north0]
z = [MAV.down0 * -1]
y = [MAV.east0]

# run time = dt*steps
step = 45

#integrate function 
for i in range(0, step):
    test.update(test.forces_moments)
    x.append(test.true_state.north)
    z.append(test.true_state.altitude)
    y.append(test.true_state.east)

#Make time array
time = [0]   
for i in range(0, step):
    t = (i+1) * dt
    time.append(t)
    
#Plots 
plt.figure()
plt.plot(time, x)
plt.title('x position over time')

plt.figure()
plt.plot(time, y)
plt.title('y position over time')

plt.figure()
plt.plot(time, z)
plt.title('z position over time')


plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()