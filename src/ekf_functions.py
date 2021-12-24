import numpy as np
import math
from copy import copy

#############################################################  
# Dynamics Functions
#############################################################  
def dynam(state_curr, dt):
    '''
    constant velocity model 
    '''
    state_next = np.zeros_like(state_curr)
    state_next[0] = state_curr[0] + state_curr[2] * dt
    state_next[1] = state_curr[1] + state_curr[3] * dt
    state_next[2:] = state_curr[2:]
    return state_next

def dynamJac(nState, dt):
    F = np.eye(nState)
    F[0, 2] = dt
    F[1, 3] = dt
    return F

#############################################################  
# Measurement Functions
#############################################################  
def uwb_meas(state_curr):
    return np.linalg.norm(state_curr[0:2])

def uwb_measJac(nState, state_curr):
    H_uwb = np.zeros((1, nState))
    range =  uwb_meas(state_curr[0:2])
    H_uwb[0, 0] = state_curr[0] / range
    H_uwb[0, 1] = state_curr[1] / range
    return H_uwb

def gps_measJac(nState, nMeas, G, A):
    '''
    Computes the double difference measurement Jacobian
    '''
    H_gps = np.zeros((nMeas, nState))
    H_gps[:, 0:2] = G
    H_gps[:, 4:] = A
    return H_gps

#############################################################  
# EKF Equations
#############################################################  
def ekf_predict(state_curr, Sig_curr, dt, Q):
    '''
    updates the state and covariance (Sig) according to the EKF 
    predict equations
    '''
    F = dynamJac(len(state_curr), dt)
    state_next = dynam(state_curr, dt)
    Sig_next = F @ Sig_curr @ (F.T) + Q

    return state_next, Sig_next

def ekf_update(state_curr, Sig_curr, y, y_hat, H, R):
    '''
    updates state and covariance according to the EKF
    update equations
    '''
    K = Sig_curr @ (H.T @ np.linalg.inv((H @ Sig_curr @ H.T + R)))
    state_next = state_curr + K @ (y - y_hat)
    Sig_next = Sig_curr - K @ H @ Sig_curr

    return state_next, Sig_next

def ekf_gnss_uwb(state_curr, Sig_curr, y, dt, w, G, A, Q, R, disp=0):
    '''
    state_curr : [r1 r2 v1 v2 n1 ... nK], state vector length K (max number of satellites visible during segment)
    Sig_curr   : Covariance [4 + K, 4 + K]
    y          : measurement vector, length 2k + w (k - num. of satellites visible at this iter., w - # UWB ranges)
                 [GPS code, GPS carrier, UWB ranges]
    dt         : time step
    w          : number of range measurements
    Q          : process noise [4 + K, 4 + K]
    R          : measurement noise [2k + w, 2k + w]

    (current version of the code requires that k and K are equal, e.g. no missing or additional measurements 
    at each time step)
    '''

    nState = len(state_curr)
    nMeas = len(y)
    if G.shape[0] // 2 != len(state_curr) - 4:
        print('Error: number of satellites visible not constant')
        return

    #- predict -----------------------
    state_t1, Sig_t1 = ekf_predict(state_curr, Sig_curr, dt, Q)

    #- update -----------------------
    # measurement jacobian [2k + w, 4 + k]
    H_uwb = uwb_measJac(nState, state_t1)
    H_gps = gps_measJac(nState, nMeas - w, G, A)
    H = np.concatenate((H_gps, H_uwb))
    meas_pred = H @ state_t1
    meas_pred[-1] = uwb_meas(state_t1)
    
    state_next, Sig_next = ekf_update(state_t1, Sig_t1, y, meas_pred, H, R)

    if disp:
        print('H', H)
        print('yhat', meas_pred)


    return state_next, Sig_next
