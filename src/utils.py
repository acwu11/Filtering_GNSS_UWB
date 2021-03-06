# This file contain multiple utility function for working with GNSS data in Python
# In all functions, "eph" is a xarray containing navigation information extracted using georinex

import numpy as np
import xarray as xr
import georinex as gr
import math
import LAMBDA

#old one
#x0 = np.array([-2634636.33395241, -4162082.25080632, 4038273.62708483]) 
#x0_lla = np.array([39.533999641572260, -122.3341865189422, 91.168144226074220]) #lat, lon, alt

#new one
x0 = np.array([-2634649.672,-4162101.155,4038292.711])
x0_lla = np.array([39.5340,-122.3342,121.1679])

def findIdxs(t, eph):
    idxs = []
    for s in eph.sv.values:
        toes = eph.sel(sv = s)['Toe']
        value = np.inf
        idx = 0
        for i in range(len(toes)):
            if not np.isnan(toes[i]) and np.abs(toes[i] - t) < value:
                idx = i
                value = np.abs(toes[i] - t)
        idxs.append(idx)
    return idxs



def timeInGPSWeek(t, date):
    '''
    Compute time in the GPS week
    '''
    dt = date-np.datetime64('1980-01-06')
    dt = dt.astype('float64')
    nWeeks = np.floor(dt / 7)
    
    return (t-np.datetime64('1980-01-06')).astype('float64') * 1e-9 - nWeeks * 7 * 24 * 60 * 60


def findOffsetsOld(eph, svs, t):
    '''
    Compute clock biases
    '''
    eph=eph.sel(sv=svs)
    if len(eph.time.values)>1:
        idxs=findIdxs(t,eph)
        i=np.arange(len(svs))
        return eph['SVclockBias'].values[idxs,i]
    return eph['SVclockBias'].values

def findOffsets(eph, svs, t):
    '''
    Compute clock biases
    '''
    eph=eph.sel(sv=svs)
    if len(eph.time.values)>1:
        idxs=findIdxs(t,eph)
        i=np.arange(len(svs))
        t0=eph['Toe'].values[idxs,i]
        bias=eph['SVclockBias'].values[idxs,i]
        drift=eph['SVclockDrift'].values[idxs,i]
        driftrate=eph['SVclockDriftRate'].values[idxs,i]
        offsets=bias+drift*(t-t0)+driftrate*(t-t0)**2
        return offsets
    else:
        t0=eph['Toe'].values
        bias=eph['SVclockBias'].values
        drift=eph['SVclockDrift'].values
        driftrate=eph['SVclockDriftRate'].values
        offsets=bias+drift*(t-t0)+driftrate*(t-t0)**2
        return offsets

def computeFlightTimes(codes, svs, eph, t):
    '''
    Compute the signal flight times at all times for all satellites
    '''
    c = 299792458
    offsets = findOffsets(eph, svs,t)
    flightTimes = codes / c - offsets
    return flightTimes 
  
def computeEmissionTimes(t, codes, svs, eph, date):
    '''
    Compute the actual emission times by substrating the time of flight
    NB: the measured code is the actual travel time *c + clock offset
    See https://gssc.esa.int/navipedia/index.php/Emission_Time_Computation
    '''
    t = timeInGPSWeek(t, date)
    ft = computeFlightTimes(codes, svs, eph,t)
    return t - ft


def solveKepler(M, e, eps=1e-15, maxiter=20):
    '''
    Solve kepler algorithm
    For GPS should converge in few iterations (even 1)
    '''
    E=M
    En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    i=1
    while np.max(np.abs(En-E))>eps and i<=maxiter:
        E=En
        En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    return En


def getPos(eph, t, flightTimes, svs):
    """
    Computes satellite position in ECEF from ephemeris data for all satellites at a given time
    See https://gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
    """
    we=7.2921151467*1e-5
    mu=3.986004418*1e14
    eph=eph.sel(sv=svs)
    if np.any(np.isnan(flightTimes)):
        print(flightTimes)
    if len(eph.time.values)>1:
        idxs=findIdxs(t,eph)
        i=np.arange(len(svs))
        Toe=eph['Toe'].values[idxs,i]
        if np.any(np.isnan(Toe)):
            print('argaga')
        M0=eph['M0'].values[idxs,i]
        sqrtA=eph['sqrtA'].values[idxs,i]
        w=eph['omega'].values[idxs,i]
        e=eph['Eccentricity'].values[idxs,i]
        i0=eph['Io'].values[idxs,i]
        omega0=eph['Omega0'].values[idxs,i]
        dN=eph['DeltaN'].values[idxs,i]
        idot=eph['IDOT'].values[idxs,i]
        omegadot=eph['OmegaDot'].values[idxs,i]
        cuc=eph['Cuc'].values[idxs,i]
        cus=eph['Cus'].values[idxs,i]
        crc=eph['Crc'].values[idxs,i]
        crs=eph['Crs'].values[idxs,i]
        cic=eph['Cic'].values[idxs,i]
        cis=eph['Cis'].values[idxs,i]
    else:
        Toe=eph['Toe'].values
        M0=eph['M0'].values
        sqrtA=eph['sqrtA'].values
        w=eph['omega'].values
        e=eph['Eccentricity'].values
        i0=eph['Io'].values
        omega0=eph['Omega0'].values
        dN=eph['DeltaN'].values
        idot=eph['IDOT'].values
        omegadot=eph['OmegaDot'].values
        cuc=eph['Cuc'].values
        cus=eph['Cus'].values
        crc=eph['Crc'].values
        crs=eph['Crs'].values
        cic=eph['Cic'].values
        cis=eph['Cis'].values
    tk=t-Toe-flightTimes
    tk+=604800*(-1*(tk>302400)+(tk<-302400))
    Mk=M0+(np.sqrt(mu)/sqrtA**3+dN)*tk
    Ek=solveKepler(Mk,e)
    nuk=np.arctan2(np.sqrt(1-e**2)*np.sin(Ek),(np.cos(Ek)-e))
    uk=w+nuk+cuc*np.cos(2*(w+nuk))+cus*np.sin(2*(w+nuk))
    rk=sqrtA**2*(1-e*np.cos(Ek))+crc*np.cos(2*(w+nuk))+crs*np.sin(2*(w+nuk))
    ik=i0+idot*tk+cic*np.cos(2*(w+nuk))+cis*np.sin(2*(w+nuk))
    ldak=omega0+(omegadot-we)*tk-we*Toe
    xk=rk*np.cos(uk)
    yk=rk*np.sin(uk)
    x=xk*np.cos(ldak)-yk*np.sin(ldak)*np.cos(ik)
    y=xk*np.sin(ldak)+yk*np.cos(ldak)*np.cos(ik)
    z=yk*np.sin(ik)
    c=299792458
    rel_cor = -2 * np.sqrt(mu) *sqrtA * e * np.sin(Ek) /(c**2)
    if np.any(np.isnan(x)):
        print('ratata')
        print(sqrtA)
    return np.block([[x],[y],[z]]).T

def computeLOS(sat_pos, x0=np.zeros(3)):
    '''
    Compute the line of sight vectors for all satellites to a given position
    '''
    
    los=sat_pos-x0
    los=los/np.linalg.norm(los,axis=1,keepdims=True)
    return los

def computeRotation(theta):
    '''
    Construct rotation matrix of angle theta
    '''
    return np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

def correctPositionOld(sat_pos, flightTimes):
    '''
    Correct satellite position for the earth rotation
    '''
    
    we=7.2921151467*1e-5
    theta=we*flightTimes
    n=len(flightTimes)
    cpos=np.zeros(sat_pos.shape)
    for i in range(n):
        cpos[i]=np.dot(computeRotation(theta[i]),sat_pos[i])
    return cpos
    
def correctPosition(sat_pos, x0=x0):
    '''
    Correct satellite position for the earth rotation
    '''
    c = 299792458
    flightTimes = np.linalg.norm(sat_pos-x0,axis=1)/c #compute approximate flight time for rotation
    we=7.2921151467*1e-5
    theta=we*flightTimes
    n=len(flightTimes)
    cpos=np.zeros(sat_pos.shape)
    for i in range(n):
        cpos[i]=np.dot(computeRotation(theta[i]),sat_pos[i])
    return cpos

def computeGeoMatrixDD(los, ref=0):
    '''
    Compute the (double difference) geometry matrix from los assuming the reference satellite is ref (index)
    '''
    G = - los
    G = G - G[ref]
    return np.delete(G, ref, axis=0)

def computeGeometry(eph, t, ft, svs, x0=np.zeros(3), ref = None, plane = None):
    '''
    Compute the goemetry matrix at a given time. If a ref index is given, compute the double difference with this reference
    '''
    sat_pos=getPos(eph,t,ft,svs)
    sat_pos=correctPosition(sat_pos,x0)
    los=computeLOS(sat_pos,x0)
    if plane:
        for i in range(len(los)):
            los[i] = ecef2enu(los[i],shift = False)

    elevation = 180*np.arccos(los[:,2])/np.pi 
    
    if ref is not None:
        G =  computeGeoMatrixDD(los, ref)
    else:
        G =  -los
    #if plane:
    #    for i in range(len(G)):
    #        G[i] = ecef2enu(G[i], shift = False)
    #    #return G[:,:-1]
    #    return G
    #else:
    #    return G
    return G
    
    
def ecef2enu(x, lat0 = x0_lla[0], lon0 = x0_lla[1], shift = True, x0 = x0):
    
    phi = np.radians(lat0)
    lda = np.radians(lon0)

    sl = np.sin(lda)
    cl = np.cos(lda)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    if shift:
        x = x-x0

    x_enu = np.zeros(3)
    x_enu[0] = -sl * x[0] + cl * x[1]
    x_enu[1] = -cl * sp * x[0] - sl * sp * x[1] + cp * x[2]
    x_enu[2] = cl * cp * x[0] + sl * cp * x[1] + sp * x[2]

    return x_enu

    
def computeDD(code1, code2, carrier1, carrier2, lda, ref=0 ):
    """
    Generate double difference code and carrier measurements
    """
    codes = code1 - code2
    codes = codes - codes[ref]
    codes = np.delete(codes, ref)
    carriers = carrier1 - carrier2
    carriers = carriers - carriers[ref]
    carriers = np.delete(carriers, ref)
    return np.concatenate([lda*carriers, codes])

def computeSigma(n, sigma_code = None, sigma_phase = None, f = 1575.42*10**6, phase_error = 0.025):
    """
    Construct a correlated noise matrix for double difference measurements
    """
    if sigma_code is None or sigma_phase is None:
        sigma_code, sigma_phase = defaultSTD(f, phase_error)
    sigma = np.eye(2*n)
    sigma[:n,:n]+=np.ones((n,n))
    sigma[:n,:n]*=2*sigma_phase**2
    sigma[n:,n:]+=np.ones((n,n))
    sigma[n:,n:]*=2*sigma_code**2
    return sigma

def defaultSTD(f = 1575.42*10**6, phase_error = 0.025):
    """
    Generate default noise values for carrier phase and code measurements
    """
    c=299792458
    lda = c/f
    sigma_phase = phase_error * lda
    sigma_code = sigma_phase * 100
    #sigma_phase = 0.1
    #sigma_code = 5
    #sigma_phase *= lda
    return sigma_code, sigma_phase

def computeSigma2(n, sigma_code = None, sigma_phase = None, f = 1575.42*10**6, phase_error = 0.025,ref=0):
    """
    Construct a correlated noise matrix for double difference measurements
    """
    if not isinstance(sigma_phase,list):
        return computeSigma(n,sigma_code,sigma_phase,f,phase_error)
    
    sigma_phase = np.array(sigma_phase)**2
    sigma_code= np.array(sigma_code)**2
    #print(sigma_code[ref])
    #sigma_phase = np.diag(sigma_phase)
    #print(sigma_phase)
    #sigma_code=np.diag(sigma_code)
    #A = np.eye(n+1)
    #A[:,ref] -= 1
    #A=np.delete(A,ref,axis=0)
    #sigma_code = np.dot(A,np.dot(sigma_code,A.T))
    #sigma_phase = np.dot(A,np.dot(sigma_phase,A.T))
    #print(sigma_code)
    #dd_sigma_code=np.zeros((2*n,2*n))
    #dd_sigma_phase=np.zeros((2*n,2*n))
    #dd_sigma_code[:n,:n]=sigma_code.copy()
    #dd_sigma_code[n:,n:]=sigma_code.copy()
    #dd_sigma_phase[:n,:n]=sigma_phase.copy()
    #dd_sigma_phase[n:,n:]=sigma_phase.copy()
    #print(dd_sigma_code)
    #A=np.zeros((n,2*n))
    #A[:,:n]=np.eye(n)
    #A[:,n:]=-np.eye(n)
    #print(np.dot(dd_sigma_code,A.T))
    #sigma_code = np.dot(A,np.dot(dd_sigma_code,A.T))
    #sigma_phase = np.dot(A,np.dot(dd_sigma_phase,A.T))
    
    # below does all the steps commented above much much quicker
    sigma_code = 2*(sigma_code[ref]*np.ones((n,n)) + np.diag(np.delete(sigma_code,ref)))
    sigma_phase = 2*(sigma_phase[ref]*np.ones((n,n)) + np.diag(np.delete(sigma_phase,ref)))
    sigma = np.eye(2*n)
    sigma[:n,:n] =sigma_phase
    sigma[n:,n:] =sigma_code
    return sigma

def sigmaFromCN0(cno, ksnr, phase_ratio):
    sigma_code = [ksnr * 10**(-snr/20) for snr in cno]
    sigma_phase = [sigma/phase_ratio for sigma in sigma_code]
    return sigma_code,sigma_phase


#############################################################  
# OTHER FUNCTIONS
#############################################################  

#- TIME SYNCING -----------------------------------------------------------
def match_times(ground_truth, obs_times):
    # match time stamps of ground truth and observation
    temp_ind = 0
    gt_inds = []
    obs_inds = []

    for i in range(len(ground_truth)):
        gt_t = ground_truth[i, 0]
        for j in range(temp_ind, len(obs_times)):
            if obs_times[j] == gt_t:
                gt_inds.append(i)
                obs_inds.append(j)
                temp_ind = j

    return gt_inds, obs_inds


def id_map(sv_lst, dim=2):
    id2ind = {}
    state_ind = 2 * dim
    for i in range(len(sv_lst)):
        id2ind[sv_lst[i]] = state_ind
        state_ind += 1
        
    return id2ind


#- ANGLE WRAPPING --------------------------------------------------------------

def wrap_angle_02pi(angle):
    if angle >= 2 * math.pi:
        angle -= 2 * math.pi
    elif angle <= 0:
        angle += 2 * math.pi
    return angle

def wrap_angle_0t360(angle):
    if angle >= 360:
        angle -= 360
    elif angle <= 0:
        angle += 360
    return angle

#- INT / POSITION FIXING ----------------------------------------------------------
def lambda_positionFix(float_solu, sigma, psi, k, lda, H, A, Qahat=None):
    '''
    Computes fixed integers using LAMBDA and corresponding fixed position

    Inputs: float_solu : float solution of integer ambiguity [# cycles]
    '''

    if Qahat is not None:
        pass
    else:
        C = A
        Qi = np.linalg.inv(sigma)
        Qhat = np.linalg.inv(np.dot(C.T, np.dot(Qi, C)))
        Qahat = (Qhat + Qhat.T) /2
    afixed, sqnorm, Ps, Qzhat, Z, nfixed, mu = LAMBDA.main(float_solu, Qahat, 1)
    
    if afixed.ndim > 1:
        afixed = afixed[:,0]

    a = psi[:k]
    b =  lda * afixed
    fixed_pos =  a - b
    fixed_pos = np.dot(np.linalg.pinv(H[:k]), fixed_pos)
    
    return fixed_pos, afixed, Ps

def round_positionFix(float_solu, psi, k, lda, H):
    '''
    float_solu : float solution of interger ambiguity [# cycles]
    '''
    around = []
    for jj in range(len(float_solu)):
        around.append(round(float_solu[jj][0]))
    around = np.array(around)

    a = psi[:k]
    b = lda * around
    round_pos = a - b
    round_pos = np.dot(np.linalg.pinv(H[:k]), round_pos)

    return round_pos, around
