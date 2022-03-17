# This file contains functions to process GNSS data from RINEX files
# Should contain in the future cleaner function (or class) to preprocess all data and data at one time step

import numpy as np
import xarray as xr
import georinex as gr
from utils import *

def loadTrajectories(name_obs1, name_obs2, name_eph):
    """
    Load rinex files into xarrays
    """
    return gr.load(name_obs1), gr.load(name_obs2), gr.load(name_eph)

def loadSavedTrajectories():
    """
    Load rinex files into xarrays
    """
    return xr.open_dataset('data/traj1.nc'),xr.open_dataset('data/traj2.nc'),xr.open_dataset('data/eph.nc')

def removeDuplicateSatellites(obs):
    '''
    Remove duplicate satellites in xarray to fix a possible bug in georinex/xarray
    '''
    idxs=[]
    svs=[]
    for i in range(len(obs.sv.values)):
        if obs.sv.values[i] not in svs:
            idxs.append(i)
            svs.append(obs.sv.values[i])
    obs=obs.isel(sv=idxs)
    return obs


def constructMeasurements(traj1, traj2, date,sort_cn0 = False):
    """
    Construct necessary data at each time step
    Inputs:
    Trajectories as xarrays loaded from rinex files with georinex
    """
    t1 = set(traj1.time.values)
    t2 = set(traj2.time.values)
    ts = sorted(list(t1.intersection(t2)))
    svs = []
    code1=[]
    code2=[]
    carrier1=[]
    carrier2=[]
    cnos=[]
    t_gps=[]
    for t in ts:
        t1_t = traj1.sel(time=t)
        if t1_t['C1C'].ndim > 1:
            code_t1 = t1_t['C1C'][0] #+ 500 * np.random.normal(0, 0.5, len(t1_t['C1C'][0])) 
        else:
            code_t1 = t1_t['C1C'] #+ 500 * np.random.normal(0, 0.5, len(t1_t['C1C'])) 
        t2_t = traj2.sel(time=t)
        if t2_t['C1C'].ndim > 1:
            code_t2 = t2_t['C1C'][0]
        else:
            code_t2 = t2_t['C1C'] + 5 * np.random.normal(0, 0.5, len(t2_t['C1C'])) 
        # sv1 = set([t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])])
        # sv2 = set([t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])])
        # sv = np.array(sorted(list(sv1.intersection(sv2))))
        sv1 = [t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])]
        sv2 = [t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])]
        sv = np.intersect1d(sv1,sv2)

        try:
            t1_t = t1_t.sel(sv=sv)
            t2_t = t2_t.sel(sv=sv)
        except:
            print(sv)

        if sort_cn0:
            if t1_t['C1C'].ndim > 1:
                order = np.argsort(t1_t['S1C'][0].values)
            else:
                order = np.argsort(t1_t['S1C'].values)
        else:
            if t1_t['C1C'].ndim > 1:
                order = np.arange(len(t1_t['S1C'][0].values))
            else:
                order = np.arange(len(t1_t['S1C'].values))
        
        if t1_t['C1C'].ndim > 1:     
            code1.append(t1_t['C1C'][0].values[order] )
            carrier1.append(t1_t['L1C'][0].values[order])
            cnos.append(t1_t['S1C'][0].values[order])
            svs.append(sv[order])
        else:
            code1.append(t1_t['C1C'].values[order] )
            carrier1.append(t1_t['L1C'].values[order])
            cnos.append(t1_t['S1C'].values[order])
            svs.append(sv[order])
        if t2_t['C1C'].ndim > 1:
            code2.append(t2_t['C1C'][0].values[order])
            carrier2.append(t2_t['L1C'][0].values[order])
        else:
            code2.append(t2_t['C1C'].values[order])
            carrier2.append(t2_t['L1C'].values[order]) 
        t_gps.append(timeInGPSWeek(t, date))   
    return t_gps, svs, code1, code2, carrier1, carrier2, cnos, ts

def prepareData(t, svs, code1, code2, carrier1, carrier2, eph, 
                plane=False, ref=0, x0=x0, f=1575.42*10**6,
                phase_error=0.025, sigma_code=None, sigma_phase=None):
    """
    Generate psi, H, A and sigma for the optimization problem.
    Works both in 2D and 3D with the plane variable
    For 2D need to add computation of H in ENU
    
    Inputs:
    t: current date in seconds in the GPS week
    svs: list of satellite in views by both recieevrs at time t
    code1(2): list for all code measurments of receiver 1 (2) in same orders as svs
    carrier1(2): same for carrier phase
    eph: xarray with rinex navigation file loaded
    plane: whether to work in local ENU frame (not yet implemented)
    ref: reference for double difference computation
    date: day (without hours/minute/seconds) of the experiment
    x0: position for geometry matrix computation
    f: measurement frequency
    phase_error: assumed error in meter in carrier phase measurements for default noise estimation
    sigma_code,sigma_phase: can be used to specify noise standard deviations
    """

    c = 299792458
    lda = c/f
    n = len(svs) -1
    ft = computeFlightTimes(code1, svs, eph, t)
    H = computeGeometry(eph, t, ft, svs, x0, ref, plane)
    psi = computeDD(code1, code2, carrier1, carrier2, lda, ref)
    A = np.zeros((2*n,n))
    A[:n] = lda*np.eye(n)
    sigma = computeSigma2(n,sigma_code,sigma_phase,f,phase_error,ref)
    return psi, H, A, sigma

def prepareAllData(ts, svs, code1, code2, carrier1, carrier2, eph, 
                plane=False,ref=0,x0=x0,f=1575.42*10**6,
                phase_error=0.05,sigma_code=None,sigma_phase=None):
    """
    Compute data at all times.
    Should probably create a class.
    """
    psis, Hs, As, sigmas = [], [], [], []
    for i in range(len(ts)):
        psi, H, A, sigma = prepareData(ts[i],svs[i],code1[i],code2[i],carrier1[i],carrier2[i],eph,
                                       plane, ref, x0, f,
                                       phase_error, sigma_code, sigma_phase)
        psis.append(psi)
        Hs.append(H)
        As.append(A)
        sigmas.append(sigma)
    return psis, Hs, As, sigmas
