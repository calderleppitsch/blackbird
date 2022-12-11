"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from UAVBook_references.tools.rotations import Euler2Quaternion, Quaternion2Euler
import UAVBook_references.parameters.aerosonde_parameters as MAV
from UAVBook_references.parameters.simulation_parameters import ts_simulation as Ts
from UAVBook_references.message_types.msg_delta import MsgDelta
import model_coef as coef

def compute_model(mav, trim_state, trim_input):
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input[0], trim_input[1], trim_input[2], trim_input[3]))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])
    rho = MAV.rho
    S = MAV.S_wing
    b = MAV.b
    c = MAV.c

    # define transfer function constants
    a_phi1 = coef.a_phi1  #-.5* rho* Va_trim**2 * S * b * MAV.C_p_p * (b/(2*Va_trim))
    a_phi2 =  coef.a_phi2 #.5* rho* Va_trim**2 * S * b * MAV.C_p_delta_a 
    a_theta1 = coef.a_theta1 #-(rho* Va_trim**2 * c * S * MAV.C_m_q * c / (2*MAV.Jy*2*Va_trim))
    a_theta2 = coef.a_theta2  #rho* Va_trim**2 * c * S * MAV.C_m_alpha / (2*MAV.Jy)
    a_theta3 = coef.a_theta3 #rho* Va_trim**2 * c * S * MAV.C_m_delta_e / (2*MAV.Jy)

    # Compute transfer function coefficients using new propulsion model
    a_V1 = coef.a_V1  # (rho*Va_trim*S/MAV.mass)*(MAV.C_D_0+(MAV.C_D_alpha*MAV.C_D_delta_e*)+(MAV.C_D_delta_e*trim_input.item(1)))+rho*MAV.S_prop/MAV.mass*MAV.C_prop*Va
    a_V2 = coef.a_V2 #  rho*MAV.S_prop/MAV.mass*MAV.C_prop*MAV.k_motor**2.*trim_input.item(1)
    a_V3 = coef.a_V3 ## MAV.gravity*np.cos(theta) # Didn't include chi_star because in trim, chi_star should be zero.

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    #x_euler = euler_state(trim_state)
    #A = 
    #B = 
    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    A_lon = coef.A_lon
    B_lon = coef.B_lon
    # extract lateral states (v, p, r, phi, psi)
    A_lat = coef.A_lat
    B_lat = coef.B_lat
    return A_lon, B_lon, A_lat, B_lat

"""
def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    x_euler = Quaternion2Euler(x_quat)
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    x_quat = Euler2Quaternion(x_euler)
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    
    f_euler_ = 
    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    A = 
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    B = 
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 
    T_eps, Q_eps = #mav._motor_thrust_torque()
    T, Q = #mav._motor_thrust_torque()
    return (T_eps - T) / eps

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 
    T_eps, Q_eps = #mav._motor_thrust_torque()
    T, Q = #mav._motor_thrust_torque()
    return (T_eps - T) / eps

"""