"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
#COMPLETE

import sys
sys.path.append('/Users/C/Dropbox/work/blackbird/UAVBook_references')
sys.path.append('/Users/C/Dropbox/work/blackbird/BLACKBIRD_Code')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion
from message_types.msg_delta import MsgDelta  
import blackbird_dynamics_v2 as dynamics

def compute_trim(mav, Va, gamma):
    # define initial state and input
    e = Euler2Quaternion(0., gamma, 0.)
    state0 = np.array([[0.],  # (0)
                       [0.],   # (1)
                       [mav._state[2]],   # Down Position
                       [Va],   # (3)  Velocity in x direction
                       [0.],    # (4)
                       [0.],    # (5)
                       [e.item(0)],    # (6) #Initial Orientation
                       [e.item(1)],    # (7)
                       [e.item(2)],    # (8)
                       [e.item(3)],    # (9)
                       [0.],    # (10)
                       [0.],    # (11)
                       [0.]     # (12)
                       ],dtype='float')
    delta0 = MsgDelta()
    x0 = np.concatenate((state0, delta0.to_array()), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma), constraints=cons, options={'ftol': 1e-10, 'disp': False})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input

def trim_objective_fun(x, mav, Va, gamma):
    x_star = x[0:13]
    e0 = x_star[6]
    e1 = x_star[7]
    e2 = x_star[8]
    e3 = x_star[9]
    normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
    x_star[6] = e0/normE
    x_star[7] = e1/normE
    x_star[8] = e2/normE
    x_star[9] = e3/normE
    delta_star = x[13:17]
    desired_trim_state_dot = np.array([[0],[0],[-Va*np.sin(gamma)],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    mav._state = x_star
    mav._update_velocity_data()
    delta = MsgDelta()
    delta.from_array(delta_star)
    forces_moments = mav._forces_moments(delta)
    f = mav._derivatives(x_star, forces_moments)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[2:13])**2.0 
    return J

#TEST TO MAKE SURE IT WORKS
"""
mav = dynamics.MavDynamics(10)
Va = 25.
gamma = 0.*np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)

print(trim_state)
"""
