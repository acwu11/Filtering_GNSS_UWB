from operator import le
import os
from textwrap import wrap
import numpy as np
import math

from scipy.sparse.linalg.dsolve.linsolve import use_solver
from preprocessing import *
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from init_covariances import *
from ekf_functions import *
import LAMBDA


np.random.seed(0)
cwd = os.getcwd()
dir_path = cwd[:-3]

###################### SETUP ######################
#- params -------------------------------------------
# measurement date in YYYY-MM-DD
date = np.datetime64("2021-08-26")

# trajectory and  data collection params
truth_frequency = 4
dt = 1 / truth_frequency
obs_freq = 2
leap_seconds = 18
x0 = np.array([-2.7011e+06,-4.292e+06, 3.8554e+06])

# other
ksnr = 200
phase_ratio = 300 	    # sigma phase = sigma code / phase_ratio
gt_xy_init = [19, 23]   # first ground truth x, y coord
f = 1575.42 * 10 ** 6
c = 299792458
lda = c / f


#- observation and grouth truth files ------------------
name_obs1 = dir_path + 'data/20210826_data/leader2.21O'
name_obs2 = dir_path + 'data/20210826_data/follower.21O' 
name_eph =  dir_path + 'data/20210826_data/leader2.21N'

ground_truth = np.loadtxt(dir_path + 'data/20210826_data/Ground_Truth_Static.csv', delimiter = ",")  # ENU
uwb_data = np.loadtxt(dir_path + 'data/20210826_data/UWB_Baseline_Static.csv', delimiter = ",")    

#- preprocess-------------------------------------------
traj1, traj2, eph = loadTrajectories(name_obs1, name_obs2, name_eph)
print('trajectories loaded')

t_gps, svs, code1, code2, carrier1, carrier2, cnos, ts = constructMeasurements(traj1, traj2, date, sort_cn0 = False)
cnos = np.array(cnos)
print('measurements constructed')


################### INITIALIZE FILTER ####################
# filtering params
common_svs = ['G08', 'G10', 'G23', 'G27', 'G32']
K = len(common_svs) - 1     # minus 1 for reference satellite
ref_ind = -1
w = 1                       # num UWB range
nState = 4 + K
nMeas = 2 * K + w   

x_pre = np.zeros((nState, 1))
x_pre[0] = 19
x_pre[1] = 25
x_pre[2] = 1
x_pre[3] = 1
x_pre[4] = -19
x_pre[5] = -6
x_pre[6] = -5
x_pre[7] = 1

P_pre, Q, R, sig_uwb, sig_gps = init_covariances(K, nState, nMeas)

################### MAIN LOOP #############################
num_sats = []

truth = []
est = []
est.append(x_pre)
error = []
covars = []

#---- match the start time between ground truth and obs. files -------
obs_start_ind = get_obs_startInd(ts, date, ground_truth)
if truth_frequency > obs_freq:
    gt_inds = np.arange(0, len(ground_truth), truth_frequency / obs_freq)
    obs_inds = np.arange(obs_start_ind, obs_start_ind + math.ceil(len(ground_truth) / (truth_frequency / obs_freq)), 1)
else:
    gt_inds = np.arange(0, len(ground_truth), 1)
    obs_inds = np.arange(obs_start_ind, obs_start_ind + (len(ground_truth)-1) * (obs_freq / truth_frequency), obs_freq / truth_frequency)

gt_inds = gt_inds.astype(int)
obs_inds = obs_inds.astype(int)
starting = 1

sigma_code = [1 for i in range(8)]
sigma_phase= [0.01 for i in range(8)]
init_n = None
print('entering loop')
# -------------------------------------------------------------------

for i in range(83, 112):
    # get appropriate indices
    gt_ind = gt_inds[i]
    obs_ind = obs_inds[i]
    
    # GROUND TRUTH
    truth_term = ground_truth[gt_ind, 1:]
    truth.append(truth_term[0:2])

    # MEASUREMENTS
    # gps
    svs_obs = svs[obs_ind]
    select_inds = np.where(np.isin(svs_obs, common_svs))
    select_inds = select_inds[0].tolist()

    new_code1 = [code1[obs_ind][i] for i in select_inds]
    new_code2 = [code2[obs_ind][i] for i in select_inds]
    new_carrier1 = [carrier1[obs_ind][i] for i in select_inds]
    new_carrier2 = [carrier2[obs_ind][i] for i in select_inds]

    sigma_code, sigma_phase = sigmaFromCN0(cnos[obs_ind], ksnr, phase_ratio)
    # psi, G, A, sigma = prepareData(t_gps[obs_ind], svs[obs_ind], code1[obs_ind], code2[obs_ind], carrier1[obs_ind], carrier2[obs_ind], eph, plane=False, ref= -1, x0=x0, f=1575.42*10**6, phase_error=0.025)
    psi, G, A, sigma = prepareData(t_gps[obs_ind], common_svs, np.array(new_code1), np.array(new_code2), np.array(new_carrier1), np.array(new_carrier2), eph, plane=False, ref= -1, x0=x0, f=1575.42*10**6, phase_error=0.025)
    k = psi.shape[0] // 2               # number of satellites visibe this time step
    H = np.zeros((2 * k, G.shape[1]))
    H[:k] = G
    H[k:] = G
    psi -= truth_term[2] * H[:, 2]
    H = H[:, :2]

    ### uncomment code below to initialize int. ambiguities in filter states
    # freq = 1575.42*10**6
    # c = 299792458
    # lda = c/freq
    # init_n = (psi[0: A.shape[0]//2] - psi[A.shape[0]//2:]) / lda
    # print(init_n)
    ######################################################################
   

    # uwb
    uwb_range = uwb_meas(truth_term[0:2]) + np.random.normal(0, sig_uwb, 1)
    y = np.reshape(np.append(psi, uwb_range), (2 * k + w, 1))

    # EKF
    print('entering filtering')
    x_next, P_next = ekf_gnss_uwb(x_pre, P_pre, y, dt, w, H, A, Q, R)

    # LAMBDA
    print('performing lambda')
    C = A
    Qi = np.linalg.inv(sigma)
    Qhat = np.linalg.inv(np.dot(C.T, np.dot(Qi, C)))
    Qahat = (Qhat + Qhat.T) /2
    afixed, sqnorm, Ps, Qzhat, Z, nfixed, mu = LAMBDA.main(x_next[4:], Qahat, 1)
    if afixed.ndim > 1:
        afixed = afixed[:,0]

    # fix position 
    a = psi[:k]
    b =  lda * afixed
    x_fixed =  a - b
    x_fixed = np.dot(np.linalg.pinv(H[:k]), x_fixed)
    x_fixed = np.reshape(x_fixed, (2, 1))

    x_next[0:2] = x_fixed
    est.append(np.ndarray.flatten(x_next[0:2]))
    covars.append(P_next)

    x_pre = copy(x_next)
    P_pre = copy(P_next)

    # stats
    err = truth_term[0:2] - np.ndarray.flatten(x_next[0:2])
    error.append(err)



print ('code done')
print ('truth', truth)
print('est', est)
print('error', error)
print('covars', covars)

# plotting
truth = np.array(truth)
est = np.array(est)
error = np.array(error)
covars = np.array(covars)

plt.figure()
plt.title('GT vs. Est')
plt.plot(truth[:, 0], truth[:, 1])
plt.plot(est[:,0], est[:,1])
plt.xlabel('East')
plt.ylabel('North')
plt.legend(['Truth', 'Est.'])

plt.figure()
plt.title('Covariance East and North')
plt.plot(covars[:,0,0])
plt.plot(covars[:,1,1])
plt.xlabel('Time')
plt.ylabel('$\sigma^{2}$')
plt.legend(['East', 'North'])

plt.figure()
plt.title('Covariance Velocity')
plt.plot(covars[:,2,2])
plt.plot(covars[:,3,3])
plt.xlabel('Time')
plt.ylabel('$\sigma^{2}$')
plt.legend(['$v_{e}$', '$v_{n}$'])


