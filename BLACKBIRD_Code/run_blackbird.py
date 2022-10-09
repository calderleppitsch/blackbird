# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:16:00 2022

This is the file to run which plots the dynamics using runge Kutta 4 integration method
"""

import numpy as np
import matplotlib.pyplot as plt
import params_blackbird as MAV
import mav_dynamics_blackbird as dynamics


#time step for integraion
dt = .1

#initialize the dynamics class
test = dynamics.MavDynamics(dt)

def forces_moments(time):
    fx = 0
    fy = 0
    fz = 0
    l = 0
    m = 0
    n = 0
    return np.array([[fx],
                    [fy],
                    [fz],
                    [l],
                    [m],
                    [n]])

#Initialize State Arrays with ICs from Parameters 
x = [MAV.north0]
z = [MAV.down0 * -1]
y = [MAV.east0]
u = [MAV.u0]
v = [MAV.v0]
w = [MAV.w0]
phi = [MAV.phi0]
theta = [MAV.theta0]
psi = [MAV.psi0]
p = [MAV.p0]
q = [MAV.q0]
r = [MAV.r0]

# run time = dt*steps
step = 100

#integrate function 
for i in range(0, step):
    test.update(forces_moments(i*dt))
    x.append(test.true_state.north)
    z.append(test.true_state.altitude)
    y.append(test.true_state.east)
    u.append(test.true_state.u)
    v.append(test.true_state.v)
    w.append(test.true_state.w)
    phi.append(test.true_state.phi)
    theta.append(test.true_state.theta)
    psi.append(test.true_state.psi)
    p.append(test.true_state.p)
    q.append(test.true_state.q)
    r.append(test.true_state.r)

#Make time array
time = [0]   
for i in range(0, step):
    t = (i+1) * dt
    time.append(t)
    
#Plots 
plt.figure()
plt.plot(time, x, label = 'x pos')
plt.plot(time, y, label = 'y pos')
plt.plot(time, z, label = 'z pos')
plt.title('Position over time')

plt.xlabel('time (s)')
plt.ylabel('distance (m)')
plt.legend()

plt.figure()
plt.plot(time, u, label = 'x vel')
plt.plot(time, v, label = 'y vel')
plt.plot(time, w, label = 'z vel')
plt.title('Velocity over time')

plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()

plt.figure()
plt.plot(time, phi, label = '$phi$')
plt.plot(time, theta, label = '$theta$')
plt.plot(time, psi, label = '$psi$')
plt.title('Rotation over time')

plt.xlabel('time (s)')
plt.ylabel('Rotation (degrees)')
plt.legend()

plt.figure()
plt.plot(time, p, label = '$\phi$ vel')
plt.plot(time, q, label = '$theta$ vel')
plt.plot(time, r, label = '$\psi$ vel')
plt.title('Rotational Velocity over time')

plt.xlabel('time (s)')
plt.ylabel('rotational velocity (deg/s)')
plt.legend()

plt.show()