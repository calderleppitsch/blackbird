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
from UAVBook_references.message_types.msg_state import MsgState
import UAVBook_references.parameters.aerosonde_parameters as MAV
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
        self._state = self._state + (time_step/6 * (k1 + 2*k2 + 2*k3 + k4))

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

    def _update_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.u = self._state.item(3)
        self.true_state.v = self._state.item(4)
        self.true_state.w = self._state.item(5)
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
