# Contains function to apply lambda algorithm to data
# Also contains code to analyze lambda if run as a script

import numpy as np
import LAMBDA
import matplotlib.pyplot as plt
from utils import *
from preprocessing import *
#from tqdm import tqdm
import xarray as xr
import georinex as gr

#ntest1=np.array([-275., -412., -424., -358., -307., -326., -296., -300., -313.])
#ntest2 = np.array([-270., -408., -431., -357., -312., -319., -298., -299., -307.])
#ntest3 = np.array([-260., -418., -421., -367., -301., -297., -306., -292., -312.])
#ntest4 = np.array([-264., -403., -416., -358., -306., -305., -300., -296., -303.])
#ntest5 = np.array([-265., -399. ,-399. ,-357. ,-299. ,-303. ,-298., -296., -303.])
#ntest6 = np.array([-266., -400., -400., -357., -299., -305., -298., -296., -304.])
#ntest = np.array([-264.-2, -397.-2, -400.-0, -358.+1, -299.+1, -306.+1, -298., -296., -304.+3])
#ntest = np.array([-264.-1, -397.-1, -400.+1, -358.+2, -299.+0, -306.+2, -298.+1, -296., -304.+2])
#ntest = np.array([-264.-2+0, -397.-2+0, -400.-0+0, -358.+1+0, -299.-1+1, -306.+1+0, -298.+0+1, -296., -304.+1+0])
ntest = np.array([-266., -399., -400., -357., -300., -305., -298., -297., -304.])
#starting index for line : 297

def floatSolution(psi,A,H, sigma):
    """
    Compute weighted least-square solution
    """
    if A is not None:
        C=np.hstack((A,H))
        Qi=np.linalg.inv(sigma)
        Qhat=np.linalg.inv(np.dot(C.T,np.dot(Qi,C)))
        Qhat = (Qhat + Qhat.T) /2
        xhat=np.dot(Qhat,np.dot(C.T,np.dot(Qi,psi)))
        return xhat,Qhat  
    else:
        C = H
        Qi=np.linalg.inv(sigma)
        Qhat=np.linalg.inv(np.dot(C.T,np.dot(Qi,C)))
        Qhat = (Qhat + Qhat.T) /2
        xhat=np.dot(Qhat,np.dot(C.T,np.dot(Qi,psi)))
        return xhat,Qhat 
    
def baseLambda(xhat,Qhat,plane = False,reduce=True):
    """
    Apply LAMBDA to the given input and return fixed integers
    """
    if reduce:
        if plane:
            xhat=xhat[:-2]
            Qhat=Qhat[:-2,:-2]
        else:
            xhat=xhat[:-3]
            Qhat=Qhat[:-3,:-3]
    #Qhat = np.eye(len(Qhat)) + np.random.normal(size=Qhat.shape)
    #Qhat = (Qhat+Qhat.T)/2
    #Qhat = np.dot(Qhat,Qhat)

    afixed,sqnorm,Ps,Qzhat,Z,nfixed,mu = LAMBDA.main(xhat,Qhat,1, 5, mu=1)
    print("Ps: ",Ps)
    #print("nfixed: ",nfixed)
    #print(afixed,sqnorm,Ps,Qzhat,Z,nfixed,mu)
    #print(afixed)
    if afixed.ndim > 1:
        return afixed[:,0]
    
    return afixed

def finalSolution(xhat, Qhat, afixed, plane = False):
    if plane:
        return xhat[-2:]-np.dot(Qhat[-2:,:-2], np.linalg.solve(Qhat[:-2,:-2], xhat[:-2]-afixed)) 
    else:
        #print('Added term: ',np.dot(Qhat[-3:,:-3], np.linalg.solve(Qhat[:-3,:-3], xhat[:-3]-afixed)))
        return xhat[-3:]-np.dot(Qhat[-3:,:-3], np.linalg.solve(Qhat[:-3,:-3], xhat[:-3]-afixed)) 

def lambdaFromData(t, svs, code1, code2, carrier1, carrier2, eph,plane=False, 
                ref=0,x0=x0,f=1575.42*10**6,
                phase_error=0.1,sigma_code=None,sigma_phase=None,afixed=None):
    """
    Apply lambda to a set of measurements
    """
    #print('code1:',code1[0],code1[-1])
    #print('carrier1:',carrier1[0],carrier1[-1])
    psi, G, A, sigma = prepareData(t, svs, code1, code2, carrier1, carrier2, eph, 
                plane, ref,x0,f,
                phase_error,sigma_code,sigma_phase)
    n = len(psi)//2
    H = np.zeros((2*n,G.shape[1]))
    H[:n] = G
    H[n:] = G
    xhat,Qhat = floatSolution(psi,A,H,sigma)
    #print ('xxx', xhat.shape)
    if afixed is None:
        #print ('kar')
        afixed = baseLambda(xhat, Qhat, plane)
    n = len(psi)//2
    c=299792458
    lda = c/f
    #x_fixed, Q_fixed = floatSolution(psi[:n]-np.dot(A[:n],afixed), None, H[:n],sigma[:n,:n])
    #x_fixed = psi[:n] - lda * afixed
    #x_fixed = np.dot(np.linalg.pinv(H[:n]), x_fixed)
    x_fixed = finalSolution(xhat, Qhat, afixed, plane)
    if plane:
        xfloat = xhat[-2:]
    else:
        xfloat = xhat[-3:]
    return afixed, x_fixed, Qhat, xfloat



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path[:-3] + 'data/'
    truth1 = np.loadtxt(dir_path + 'Trajectory1/trajectory1.csv', delimiter = ",")
    truth2 = np.loadtxt(dir_path + 'Trajectory2/trajectory2.csv', delimiter=',')
    name_obs1 = dir_path + 'Trajectory1/traj1.21O'
    name_obs2 = dir_path + 'Trajectory2/traj2.21O'
    name_eph =  dir_path +'Trajectory1/traj1.21N'
    date = np.datetime64("2020-02-06")

    #truth1 = np.loadtxt('data/Line1/line1.csv', delimiter = ",")
    #truth2 = np.loadtxt('data/Line2/line2.csv', delimiter=',')
    #plt.figure()
    #plt.plot(truth1[:,1]-truth2[:,1])
    #plt.show()


    traj1, traj2, eph = loadTrajectories(name_obs1, name_obs2, name_eph)
    t_gps, svs, code1, code2, carrier1, carrier2, ts = constructMeasurements(traj1, traj2, date)
    c=299792458
    f=1575.42*10**6
    lda = c/f
    '''
    to_plot_d=[]
    to_plot_r=[]
    to_plot_p=[]
    for i in range(len(code1)):
        psi = computeDD(code1[i],code2[i], carrier1[i], carrier2[i],lda)
        n = len(psi)//2
        to_plot_d.append(psi[5]-psi[n+5])
        to_plot_p.append(psi[5])
        to_plot_r.append(psi[n+5])
    #plt.plot([code1[i][0] - lda * carrier1[i][0] for i in range(len(code1))])
    #plt.plot([code1[i][1] - lda * carrier1[i][1] for i in range(len(code1))])
    #plt.plot([code1[i][2] - lda * carrier1[i][2] for i in range(len(code1))])
    plt.figure()
    plt.plot(to_plot_d)
    plt.figure()
    plt.plot(to_plot_r)
    plt.figure()
    plt.plot(to_plot_p)
    plt.show()
    '''
    #print('sv',svs[0],svs[-1])
    #print('t_gps', t_gps[0], t_gps[-1])
    xfixed = []
    nfixed = []
    truth=[]
    xfloats =[]
    carrier_code_res=[]
    inds=[]
    #ind = 682
    dds=[]
    #for i in range(2):
    for i in range(len(ts)):
        afixed, x_fixed, Q_fixed, xfloat= lambdaFromData(t_gps[i], svs[i], code1[i], code2[i], carrier1[i], carrier2[i], eph, plane=False, afixed=ntest)
        dd = computeDD(code1[i], code2[i], carrier1[i], carrier2[i], lda)
        
        n = len(afixed)
        dds.append((dd[:n] - dd[n:])/lda)
        carrier_code_res.append(dd[:n] - lda*afixed - dd[n:])
        xfixed.append(x_fixed)
        nfixed.append(afixed)
        xfloats.append(xfloat)
        ind = int(round((ts[i] - np.datetime64('2020-02-06T15:06:10')).astype('float')*1e-8))
        #truth_idx = int((ts[i] - np.datetime64('2020-02-06T15:06:10'))*1e-8)
        truth_idx=ind
        #ind+=1
        inds.append(truth_idx)
        #truth.append(ecef2enu(truth1[truth_idx,1:],shift= False)-ecef2enu(truth2[truth_idx,1:], shift= False))
        truth.append(truth1[truth_idx,1:]-truth2[truth_idx,1:])
    xfixed = np.array(xfixed)
    xfloats=np.array(xfloats)
    truth = np.array(truth)
    nfixed=np.array(nfixed)
    #carrier_code_res = np.array(carrier_code_res)
    #print(np.round(np.average(nfixed,axis=0)))
    #print(inds)
    #dds = np.array(dds)
    #print(np.round(np.average(dds,axis=0)))
    print('Avg error: ', np.average(np.linalg.norm(truth[:,:]-xfixed,axis=1)))

    # print(truth[:10])
    # print(xfixed[:10])
    # print(xfloats[:10])
    # print(nfixed[:10])
    # #print(nfixed[308])
    # #print(carrier_code_res[:10])
    
    # plt.figure()
    # plt.plot(truth[:,0])
    # f=1575.42*10**6
    # c=299792458
    # lda=c/f
    # plt.plot(xfixed[:,0])
    # plt.plot(xfloats[:,0])
    # plt.legend(['Truth ECEF x coordinate','Lambda  ECEF x coordinate','Float  ECEF x coordinate'])
    # plt.figure()
    # plt.plot(truth[:,1])
    # plt.plot(xfixed[:,1])
    # plt.plot(xfloats[:,1])
    # plt.legend(['Truth ECEF y coordinate','Lambda  ECEF x coordinate','Float  ECEF y coordinate'])
    # #plt.figure()
    # #plt.plot(truth[:,2])
    # #plt.plot(xfloats[:,2])
    # #plt.legend(['Truth ECEF z coordinate','Float  ECEF z coordinate'])
    # #plt.ylim([140,160])
    # #plt.savefig('floatx.pdf')
    # #plt.figure()
    # #plt.plot(truth[:,0],truth[:,1])
    # #plt.plot(xfixed[:,0],xfixed[:,1])
    # #plt.plot(xfloats[:,0],xfloats[:,1])
    # plt.figure()
    # plt.plot(np.linalg.norm(truth[:,:]-xfixed,axis=1),'-*')
    # plt.plot(np.linalg.norm(truth[:,:]-xfloats,axis=1))
    # plt.legend(['lambda error', 'float error'])
    # #plt.legend(['Error between ground truth and float solution'])
    # #plt.savefig('floaterror.pdf')
    # plt.figure()
    # plt.plot(truth[:,0]-xfixed[:,0],'-*')
    # plt.plot(truth[:,0]-xfloats[:,0])
    # plt.legend(['Lambda x error', 'Float x error'])
    # plt.ylim([-10,10])
    # #plt.figure()
    # #plt.plot(np.average(carrier_code_res, axis=1))
    # #plt.legend(['lda*carrier - code after fix'])
    # plt.show()

    plt.figure()
    plt.plot(truth[:,0]-xfixed[:,0],'-*')
    plt.legend(['Lambda x error'])
    plt.ylim([-0.005,0.015])
    #plt.figure()
    #plt.plot(np.average(carrier_code_res, axis=1))
    #plt.legend(['lda*carrier - code after fix'])
    #plt.show()
