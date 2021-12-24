import numpy as np

def init_covariances(k, nState, nMeas):

    # State Covariance
    sig_pos = 5
    sig_vel = 2
    sig_intamb = 5
    P = np.zeros((nState, nState))
    P[0:2, 0:2] = sig_pos ** 2 * np.eye(2)
    P[2:4, 2:4] = sig_vel ** 2 * np.eye(2)
    P[4:, 4:] = sig_intamb ** 2 * np.eye(k)
    
    # Dynamics Process Noise
    sig_east = 1 # [m]
    sig_north = 1 # [m]
    sig_vel = 0.5  # [m / s]
    sig_intamb = 1    
    Q = np.zeros((nState, nState))
    Q[0, 0] = sig_east ** 2
    Q[1, 1] = sig_north ** 2
    Q[2:4, 2:4] = sig_vel ** 2 * np.eye(2)
    Q[4:, 4:] = sig_intamb ** 2 * np.eye(k)

    # Measurement Noise
    sig_uwb = 5 # [m]
    sig_gps = 2 # [m]
    R = np.zeros((nMeas, nMeas))
    R[0:2*k, 0:2*k] = sig_gps ** 2 * np.eye(2*k)
    R[2*k:, 2*k:] = sig_uwb ** 2 * np.eye(nMeas - 2*k)

    return P, Q, R, sig_uwb, sig_gps