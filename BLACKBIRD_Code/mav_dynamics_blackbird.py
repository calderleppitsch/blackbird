"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/17/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('/Users/C/Dropbox/work/blackbird')
import numpy as np

# load message types
from msg_state_blackbird import MsgState
import params_blackbird as MAV
from UAVBook_references.tools.rotations import Quaternion2Euler, Quaternion2Rotation


class MavDynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        self.true_state = MsgState()
        #self.forces_moments = np.array([MAV.fx, MAV.fy, MAV.fz, MAV.l, MAV.m, MAV.n])

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics. 
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state = self._state + ((time_step/6) * (k1 + 2*k2 + 2*k3 + k4))

        #round off to zero if there is small error
        for i in range(0, 12):
            if self._state[i][0] < .000001 and self._state[i][0] > -.000001:
                self._state[i][0] = 0.

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # north = state.item(0)
        # east = state.item(1)
        # down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        quaternion = np.transpose(np.array([e0, e1, e2, e3]))

        phi, theta, psi = Quaternion2Euler(quaternion)
        Rotation = Quaternion2Rotation(quaternion)
        v_b = np.transpose(np.array([u, v, w]))
     
        # position kinematics
        pos_dot = Rotation @ v_b
        
        north_dot = pos_dot.item(0)
        east_dot = pos_dot.item(1)
        down_dot = pos_dot.item(2)
    
        # position dynamics
        u_dot = (r*v - q*w)+(fx/(MAV.mass))
        v_dot = (p*w - r*u)+(fy/(MAV.mass))
        w_dot = (q*u - p*v)+(fz/(MAV.mass))

        # rotational kinematics
        R = np.array([[0, -p, -q, -r],
                      [p, 0, r, -q],
                      [q, -r, 0, p],
                      [r, q, -p, 0]])
        e_dot = .5 * (R @ quaternion)
        
        e0_dot = e_dot.item(0)
        e1_dot = e_dot.item(1)
        e2_dot = e_dot.item(2)
        e3_dot = e_dot.item(3)

        # rotatonal dynamics
        # moments = np.array([l,m,n])
        # w = np.array([p,q,r])
        # J = np.array([[MAV.Jx, 0, -MAV.Jxz],
        #               [0, MAV.Jy, 0],
        #               [-MAV.Jxz, 0, MAV.Jx]])
        # omega_dot = np.linalg.inv(J) @ (np.cross(-w, (J @ np.transpose(w))) + moments)
        # p_dot = omega_dot[0]
        # q_dot = omega_dot[1]
        # r_dot = omega_dot[2]
        p_dot = MAV.gamma1*p*q-MAV.gamma2*q*r+MAV.gamma3*l+MAV.gamma4*n
        q_dot = MAV.gamma5*p*r-MAV.gamma6*(p**2-r**2)+(m/MAV.Jy)
        r_dot = MAV.gamma7*p*q-MAV.gamma1*q*r+MAV.gamma4*l+MAV.gamma8*n
        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    # def _update_velocity_data(self, wind=np.zeros((6,1))):
    #     steady_state = wind[0:3]
    #     gust = wind[3:6]
    #     # convert wind vector from world to body frame and add gust
    #     wind_body_frame =
    #     # velocity vector relative to the airmass
    #     v_air = 
    #     ur = 
    #     vr = 
    #     wr = 
    #     # compute airspeed
    #     self._Va = 
    #     # compute angle of attack
    #     if ur == 0:
    #         self._alpha = 
    #     else:
    #         self._alpha = 
    #     # compute sideslip angle
    #     if self._Va == 0:
    #         self._beta = 
    #     else:
    #         self._beta = 

    # def _forces_moments(self, delta):
    #     """
    #     return the forces on the UAV based on the state, wind, and control surfaces
    #     :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
    #     :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
    #     """
    #     phi, theta, psi = Quaternion2Euler(self._state[6:10])
    #     p = self._state.item(10)
    #     q = self._state.item(11)
    #     r = self._state.item(12)

    #     # compute gravitaional forces
    #     f_g = 

    #     # compute Lift and Drag coefficients
    #     CL =
    #     CD =
    #     # compute Lift and Drag Forces
    #     F_lift = 
    #     F_drag = 

    #     #compute propeller thrust and torque
    #     thrust_prop, torque_prop = #self._motor_thrust_torque()

    #     # compute longitudinal forces in body frame
    #     fx = 
    #     fz = 

    #     # compute lateral forces in body frame
    #     fy = 

    #     # compute logitudinal torque in body frame
    #     My = 
    #     # compute lateral torques in body frame
    #     Mx = 
    #     Mz = 

    #     self._forces[0] = fx
    #     self._forces[1] = fy
    #     self._forces[2] = fz
    #     return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # def _motor_thrust_torque(self, Va, delta_t):
    #     # compute thrust and torque due to propeller  (See addendum by McLain)
    #     # map delta_t throttle command(0 to 1) into motor input voltage
    #     V_in = 

    #     # Angular speed of propeller
    #     Omega_p = 

    #     # thrust and torque due to propeller
    #     thrust_prop = 
    #     torque_prop =  
    #     return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.u = self._state.item(3)
        self.true_state.v = self._state.item(4)
        self.true_state.w = self._state.item(5)
        # self.true_state.Va = self._Va
        # self.true_state.alpha = self._alpha
        # self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        # self.true_state.Vg = np.linalg.norm(pdot)
        # self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        # self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        # self.true_state.wn = self._wind.item(0)
        # self.true_state.we = self._wind.item(1)
        













