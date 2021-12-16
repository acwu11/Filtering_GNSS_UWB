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


np.random.seed(0)
dir_path = '/content/gdrive/MyDrive/TRI_KF/'

###################### SETUP ######################

# define the measurement date in YYYY-MM-DD
date = np.datetime64("2021-08-26")

# params
truth_frequency = 4
obs_freq = 2
leap_seconds = 18
x0 = np.array([-2.7011e+06,-4.292e+06, 3.8554e+06])

# observation and grouth truth files
name_obs1 = dir_path + '20210826_data/leader2.21O'
name_obs2 = dir_path + '20210826_data/follower.21O' 
name_eph =  dir_path + '20210826_data/leader2.21N'

ground_truth = np.loadtxt(dir_path + '20210826_data/Ground_Truth_Static.csv', delimiter = ",")  # ENU
uwb_data = np.loadtxt(dir_path + '20210826_data/UWB_Baseline_Static.csv', delimiter = ",")    

# preprocess
traj1, traj2, eph = loadTrajectories(name_obs1, name_obs2, name_eph)
print('trajectories loaded')

t_gps, svs, code1, code2, carrier1, carrier2, cnos, ts = constructMeasurements(traj1, traj2, date, sort_cn0 = False)
cnos = np.array(cnos)
print('measurements constructed')


#################### COMPARE GT AND UWB ##############
gt_ranges = []

err_gt_uwb_range = []

for i in range(0, 250):
    gt_range = np.linalg.norm(ground_truth[i, 1:])
    gt_ranges.append(gt_range)
    err_gt_uwb_range.append(gt_range - uwb_data[i, 0])

# range comparison
plt.figure()
plt.plot(gt_ranges)
plt.plot(uwb_data[:,0])
plt.legend(['gt', 'uwb'])
plt.ylabel('Range [m]')
plt.xlabel('Time [s]')
plt.savefig('uwb_error_gt.png')
plt.show()
print("Ranging Avg. Error: {avg}".format(avg=np.mean(err_gt_uwb_range)))

###################### ALGORITHM ######################

ksnr = 200
phase_ratio = 300 	    # sigma phase = sigma code / phase_ratio
gt_xy_init = [19, 23]   # first ground truth x, y coord


f = 1575.42 * 10 ** 6
c = 299792458
lda = c / f

check_num_sats = []


sigma_code = [1 for i in range(8)]
sigma_phase= [0.01 for i in range(8)]
init_n = None
print('entering loop')


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

truth = []

for i in range(0, 30):
    gt_ind = gt_inds[i]
    obs_ind = obs_inds[i]
    
    truth_term = ground_truth[gt_ind, 1:]
    truth.append(truth_term[0:2])

    sigma_code, sigma_phase = sigmaFromCN0(cnos[obs_ind], ksnr, phase_ratio)
    psi, G, A, sigma = prepareData(t_gps[obs_ind], svs[obs_ind], code1[obs_ind], code2[obs_ind], carrier1[obs_ind], carrier2[obs_ind], eph, plane=True, ref= -1, x0=x0, f=1575.42*10**6, phase_error=0.025)
    # print ('SV',svs[obs_ind])
    # print('psi',psi.shape)
    # print('G',G.shape)
    # print ('A',A.shape)
    n_new = psi.shape[0] // 2
    H = np.zeros((2 * n_new, G.shape[1]))
    H[:n_new] = G
    H[n_new:] = G
    psi -= truth_term[2] * H[:, 2]
    H = H[:, :2]

print ('code done')
print ('truth', truth)


