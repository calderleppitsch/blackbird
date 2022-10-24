import numpy as np
import matplotlib.pyplot as plt
import params_blackbird as MAV
import blackbird_dynamics_v1 as dynamics


#time step for integration
dt = .001

#initialize the dynamics class
test = dynamics.MavDynamics(dt)

def forces_moments(phi, theta, psi, time):
    fx = -MAV.mass*9.8*np.sin((theta))
    fy = MAV.mass*9.8*np.sin((phi))*np.cos((theta))
    fz = MAV.mass*9.8*np.cos((phi))*np.cos((theta))
    l = 0.
    m = 0.
    n = 0.
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
step = 200

#integrate function 
#for i in range(0, step):
i = 0
while(z[i] >= 0):
    test.update(forces_moments(phi[i],theta[i],psi[i],i*dt))
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
    i = i+1

#Make time array
time = [0]   
for n in range(0, i):
    t = (n+1) * dt
    time.append(t)
    
# Plots 
# plt.figure()
# plt.plot(time, x, label = 'x pos')
# plt.plot(time, y, label = 'y pos')
# plt.plot(time, z, label = 'z pos')
# plt.title('Position over time in inertial frame')

# plt.xlabel('time (s)')
# plt.ylabel('distance (m)')
# plt.legend()

# plt.figure()
# plt.plot(time, u, label = 'u')
# plt.plot(time, v, label = 'v')
# plt.plot(time, w, label = 'w')
# plt.title('Velocity in body frame')

# plt.xlabel('time (s)')
# plt.ylabel('velocity (m/s)')
# plt.legend()

# plt.figure()
# plt.plot(time, phi, label = chr(966))
# plt.plot(time, theta, label = chr(952))
# plt.plot(time, psi, label = chr(968))
# plt.title('Rotation over time in inertial frame')
# plt.xlabel('time (s)')
# plt.ylabel('Rotation (degrees)')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(time, p, label = '$\phi$ vel')
# plt.plot(time, q, label = '$theta$ vel')
# plt.plot(time, r, label = '$\psi$ vel')
# plt.title('Rotational Velocity Rates')

# plt.xlabel('time (s)')
# plt.ylabel('rotational velocity (deg/s)')
# plt.legend()

# plt.figure()
# plt.plot(x, z)
# plt.title('Flight Path in XZ plane')
# plt.xlabel('Horizontal Distance [m]')
# plt.ylabel('Altitude [m]')
# plt.legend()

plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(x, y, z, 'green')
ax.set_zlabel("Altitude [m]")
ax.set_xlabel("X Distance [m]")
ax.set_ylabel("Y Distance [m]")
plt.title('Position over time in inertial frame')
plt.show()