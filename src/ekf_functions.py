import numpy as np
from copy import copy
from preprocessing import prepareData

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
    Inputs:     nState      number of states in state vector
                nMeas       number of elements in meas. vector
                G           geometry matrix (pseudorange)
                A           wavelength matrix (carrier phase)
    '''
    
    H_gps = np.zeros((nMeas, nState))
    H_gps[:, 0:2] = G
    H_gps[:, 4:] = A
    return H_gps

#############################################################  
# Basic EKF Equations
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

#############################################################
# Variations
#############################################################

def ekf_gnss(state_curr, Sig_curr, y, dt, k, G, A, Q, R, rescale=False):
    nState = len(state_curr)
    nMeas = len(y)
    if G.shape[0] // 2 != len(state_curr) - 4:
        print('Error in EKF GNSS: number of satellites visible not constant')
        return

    meas_pred = np.zeros((nMeas, 1))

    #- predict -----------------------
    state_t1, Sig_t1 = ekf_predict(state_curr, Sig_curr, dt, Q)

    #- update -----------------------
    # measurement jacobian [2k, 4 + k]
    H = gps_measJac(nState, nMeas, G, A)

    if rescale:
        state_t2 = np.zeros_like(state_t1)
        state_t2[0:4] = state_t1[0:4]
        state_t2[4:] = state_t1[4:] / 0.19
        meas_pred = H @ state_t2
    else:
        state_t2 = copy(state_t1)
        meas_pred = H @ state_t2
    
    resid = y - meas_pred
    state_next, Sig_next = ekf_update(state_t2, Sig_t1, y, meas_pred, H, R)

    if rescale:
        for i in range(k):
            state_next[4 + i] = state_next[4 + i] * 0.19


    return state_next, Sig_next, resid

def ekf_gnss_uwb(state_curr, Sig_curr, y, dt, k, w, G, A, Q, R, rescale=False):
    '''
    state_curr : [r1 r2 v1 v2 n1 ... nK], state vector length K (max number of satellites visible during segment) integer ambiguities scaled * 0.19
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
        print('Error in EKF GNSS UWB: number of satellites visible not constant')
        return

    meas_pred = np.zeros((nMeas, 1))

    #- predict -----------------------
    state_t1, Sig_t1 = ekf_predict(state_curr, Sig_curr, dt, Q)

    #- update -----------------------
    # measurement jacobian [2k + w, 4 + k]
    H_uwb = uwb_measJac(nState, state_t1)
    H_gps = gps_measJac(nState, nMeas - w, G, A)
    H = np.concatenate((H_gps, H_uwb))

    if rescale:
        state_t2 = np.zeros_like(state_t1)
        state_t2[0:4] = state_t1[0:4]
        state_t2[4:] = state_t1[4:] / 0.19
        meas_pred = H @ state_t2
        meas_pred[-1] = uwb_meas(state_t1)
    else:
        state_t2 = copy(state_t1)
        meas_pred = H @ state_t1
        meas_pred[-1] = uwb_meas(state_t1)
    
    resid = y - meas_pred
    state_next, Sig_next = ekf_update(state_t2, Sig_t1, y, meas_pred, H, R)

    if rescale:
        for i in range(k):
            state_next[4 + i] = state_next[4 + i] * 0.19


    return state_next, Sig_next, resid

def ekf_gnss_uwb_mm(state_curr, Sig_curr, y, dt, k, w, G, A, Q, R, svs_obs, sv2id, rescale=False):
    '''
    EKF for GNSS + UWB tight coupling with mismatched measurement vector depending on satellite visibility at each time
    step in the trajectory.
    '''
    nState = len(state_curr)
    nMeas = len(y)

    #- predict -----------------------
    state_t1, Sig_t1 = ekf_predict(state_curr, Sig_curr, dt, Q)

    #- update -----------------------
    # measurement jacobian [2k + w, 4 + k]
    H_uwb = uwb_measJac(nState, state_t1)
    H_gps = gps_measJac(nState, nMeas - w, G, A)
    H = np.concatenate((H_gps, H_uwb))
    meas_pred = H @ state_t1
    resid = y - meas_pred

    


    

#######################################################################
# INITIALIZATIONS
#######################################################################
def get_init_inds(ground_truth, seg_start_ind, truth_term, svs, common_svs, gt_inds, obs_inds, ref_ind, K, t_gps, code1, code2, carrier1, carrier2, eph, x0):
    '''
    Returns initialization for carrier phase integer ambiguities
    '''
    
    # get appropriate indices
    # gt_ind = gt_inds[seg_start_ind]
    # obs_ind = obs_inds[seg_start_ind]
    # if ground_truth[gt_ind, 0] != t_gps[obs_ind]:
    #     print('Error: Misaligned timestamps.')
    
    ### for simulated traj ####
    obs_ind = seg_start_ind
    ###########################

    #- MEASUREMENTS -------------------------------------------------------------------------
    # gps
    svs_obs = svs[obs_ind]
    select_inds = np.where(np.isin(svs_obs, common_svs))
    select_inds = select_inds[0].tolist()

    select_code1 = [code1[obs_ind][i] for i in select_inds]
    select_code2 = [code2[obs_ind][i] for i in select_inds]
    select_carrier1 = [carrier1[obs_ind][i] for i in select_inds]
    select_carrier2 = [carrier2[obs_ind][i] for i in select_inds]

    psi, G, A, sigma = prepareData(t_gps[obs_ind], common_svs, np.array(select_code1), np.array(select_code2), np.array(select_carrier1), np.array(select_carrier2), eph, plane=False, ref=ref_ind, x0=x0, f=1575.42*10**6, phase_error=0.025)
    k = psi.shape[0] // 2               # number of DD measurements in this time step
    if k != K:
        print('Error: Satellite number incorrect')
    H = np.zeros((2 * k, G.shape[1]))
    H[:k] = G
    H[k:] = G
    psi -= truth_term[2] * H[:, 2]
    H = H[:, :2]

    freq = 1575.42*10**6
    c = 299792458
    lda = c/freq
    init_n = (psi[0: A.shape[0]//2] - psi[A.shape[0]//2:]) / lda
    init_n = [round(x) for x in init_n]

    return np.array(init_n)

def init_covariances(k, nState, nMeas, disp=False):

    # State Covariance
    sig_state_e = 3                 # [m]
    sig_state_n = 3                 # [m]
    sig_state_ve = 5                # [m / s]
    sig_state_vn = 5                # [m / s]
    sig_state_int = 50              # [m] number of cycles # CONVERGED

    P = np.zeros((nState, nState))
    P[0, 0] = sig_state_e ** 2
    P[1, 1] = sig_state_n ** 2
    P[2, 2] = sig_state_ve ** 2
    P[3, 3] = sig_state_vn ** 2
    P[4:, 4:] = sig_state_int ** 2 * np.eye(k)
    
    # Dynamics Process Noise
    sig_east = 5                  # [m]
    sig_north = 5                 # [m]
    sig_vel_e = 10                # [m / s]
    sig_vel_n = 10
    sig_intamb = 2

    Q = np.zeros((nState, nState))
    Q[0, 0] = sig_east ** 2
    Q[1, 1] = sig_north ** 2
    Q[2, 2] = sig_vel_e ** 2
    Q[3, 3] = sig_vel_n ** 2
    Q[4:, 4:] = sig_intamb ** 2 * np.eye(k)

    # Measurement Noise
    sig_uwb = 1                    # [m]
    sig_gps_phi = 0.01             # [m]
    sig_gps_rho = 10               # [m]
     
    R = np.zeros((nMeas, nMeas))
    R[0:k, 0:k] = sig_gps_phi ** 2 * np.eye(k)
    R[k:2*k, k:2*k] = sig_gps_rho ** 2 * np.eye(k)
    R[2*k:, 2*k:] = sig_uwb ** 2 * np.eye(nMeas - 2*k)
    R_gps = R[0:2*k, 0:2*k]

    if disp:
        print('P: sig E [{sige}], sig N [{sign}], sig VE [{sigve}], sig VN [{sigvn}], sig Int [{sigint}]'.format(sige=sig_state_e, sign=sig_state_n, sigve=sig_state_ve, sigvn=sig_state_vn, sigint=sig_state_int))
        print('Q: sig E [{sige}], sig N [{sign}], sig VE [{sigve}], sig VN [{sigvn}], sig Int [{sigint}]'.format(sige=sig_east, sign=sig_north, sigve=sig_vel_e, sigvn=sig_vel_n, sigint=sig_intamb))
        print('R: sig UWB [{uwb}], sig Phi [{phi}], sig Rho [{rho}]'.format(uwb=sig_uwb, phi=sig_gps_phi, rho=sig_gps_rho))

    return P, Q, R, R_gps, sig_uwb