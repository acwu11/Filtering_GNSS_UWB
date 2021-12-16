# This file contains functions to process GNSS data from RINEX files
# Should contain in the future cleaner function (or class) to preprocess all data and data at one time step

import numpy as np
import xarray as xr
import georinex as gr
from utils import *
import os

#define the measurement date in YYYY-MM-DD
# date = np.datetime64("2020-02-06")
# #x0 = np.array([-2634636.33395241, -4162082.25080632, 4038273.62708483])
# x0 = np.array([-2634636.33074538, -4162082.2465598, 4038273.58471636])
# name_obs1 = 'data/Trajectory1/traj1.21O'
# name_obs2 = 'data/Trajectory2/traj2.21O'
# name_eph = 'data/Trajectory1/traj1.21N'
#name_obs1 = 'data/Line1/line1.21O'
#name_obs2 = 'data/Line2/line2.21O'
#name_eph = 'data/Line1/line1.21N'

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
        #print(t)
        t1_t = traj1.sel(time=t)
        if t1_t['C1C'].ndim > 1:
            code_t1 = t1_t['C1C'][0] #+ 500 * np.random.normal(0, 0.5, len(t1_t['C1C'][0])) 
        else:
            #print ('alls')
            code_t1 = t1_t['C1C'] #+ 500 * np.random.normal(0, 0.5, len(t1_t['C1C'])) 
        t2_t = traj2.sel(time=t)
        if t2_t['C1C'].ndim > 1:
            code_t2 = t2_t['C1C'][0]
        else:
            #print ('alls')
            code_t2 = t2_t['C1C'] + 5 * np.random.normal(0, 0.5, len(t2_t['C1C'])) 
        # sv1 = set([t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])])
        # sv2 = set([t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])])
        # sv = np.array(sorted(list(sv1.intersection(sv2))))
        sv1 = [t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])]
        sv2 = [t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])]
        sv = np.intersect1d(sv1,sv2)
        #svs.append(sv)
        try:
            t1_t = t1_t.sel(sv=sv)
            t2_t = t2_t.sel(sv=sv)
        except:
            print(sv)
        #print(t1_t['S1C'])
        #print(t2_t['S1C'])

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
            #print ('this')
            #k = len(t1_t['C1C'][0].values[order])
            #noise = 50 * np.random.normal(0, 0.5, 12)        
            code1.append(t1_t['C1C'][0].values[order] )
            carrier1.append(t1_t['L1C'][0].values[order])
            cnos.append(t1_t['S1C'][0].values[order])
            svs.append(sv[order])
        else:
            #print (t1_t['C1C'].values[order])
            # k = len(t1_t['C1C'][0].values[order])
            # noise = 50 * np.random.normal(0, 0.5, k)
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
                plane=False,ref=0,x0=x0,f=1575.42*10**6,
                phase_error=0.025,sigma_code=None,sigma_phase=None):
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
    c=299792458
    lda = c/f
    n=len(svs) -1
    #t = timeInGPSWeek(t, date)
    ft = computeFlightTimes(code1, svs, eph,t)
    #print(ft.shape)
    H = computeGeometry(eph, t, ft, svs, x0, ref, plane)
        
    psi = computeDD(code1, code2, carrier1, carrier2, lda, ref)
    #print ('psi', psi)
    
    A = np.zeros((2*n,n))
    A[:n]=lda*np.eye(n)
    sigma= computeSigma2(n,sigma_code,sigma_phase,f,phase_error,ref)
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

#print(gr.load(name_obs2)['L1C'])    
#print(gr.load('data/200206_224210_sec.20O')['L1'])    
#traj1, traj2, eph = loadTrajectories()
#print(traj1)
#print(traj2)
#print(eph)
#t_gps, svs, code1, code2, carrier1, carrier2, ts = constructMeasurements(traj1,traj2)
#i = 0
#while len(svs[i]) < 13:
#    i+=1
#print(i)
#print(len(t_gps))
#print(np.max(np.array(ts[1:])-np.array(ts[:-1])).astype(float)*1e-9)
#traj1.to_netcdf('data/traj1.nc')
#traj2.to_netcdf('data/traj2.nc')
#eph.to_netcdf('data/eph.nc')

#traj1,traj2,eph = loadSavedTrajectories()
#print(traj1['C1C'])