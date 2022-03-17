import math
import numpy as np
from scipy.optimize import minimize


def objective(x, psi, H, A, sigma, constraint_bounds, p, h, baseline_uwb):
    HA = np.hstack([H,A])
    HAx = np.dot(HA, x)
    c = np.dot((psi - HAx).T,np.dot(np.linalg.inv(sigma), psi - HAx))
    c += barrier_function2(constraint_bounds, x[0], x[1], p, h, baseline_uwb)
    return c

def unconstrained_optimization(psi, H, A, sigma, constraint_bounds, p, h, baseline_uwb, gt_init, init_n=None):
    two_n = A.shape[0]
    x0 = np.zeros(2 + (two_n//2))

    # initialize with first ground truth estimated position and integers 
    x0[0] = gt_init[0]
    x0[1] = gt_init[1]
    x0[2:] = np.dot(np.linalg.pinv(A), psi - np.dot(H, x0[0:2])) 

    # check the method ...
    result = minimize(lambda x: objective(x, psi, H, A, sigma, constraint_bounds, p, h, baseline_uwb), x0, method='SLSQP', tol=1e-11, jac=None, options={'ftol': 1e-13, 'disp': False})
    soln = result.x  # outputs the range, angle, N

    return soln


def barrier_function(type, constraint_bounds, var, p, h, baseline_uwb):
    '''
    Returns two barrier functions, one for angle, one for length. 
    Inputs: 	type  - barrier function type
                constraint_bounds - current bounds
                var   -  length / angle (depending on type)
    '''

    if type == 'length':
        lAB_min = baseline_uwb[0] - constraint_bounds[0]
        lAB_max = baseline_uwb[0] + constraint_bounds[1]

        if var < lAB_min-2:
             lABx = p * ((lAB_min - var) ** h)
        elif var > lAB_max+2:
             lABx = p * ((var - lAB_max) ** h)
        elif var < lAB_min:
            lABx = p * ((lAB_min - var))
        else:
            lABx = p * ((var - lAB_max))
        return lABx

    else:
        phiAB_min = baseline_uwb[1] - constraint_bounds[0]
        phiAB_max = baseline_uwb[1] + constraint_bounds[1]

        if var < phiAB_min - 0.001:
            phiABx = p * ((phiAB_min - var)**h )
        elif var > phiAB_max + 0.001:
            phiABx = p * ((var - phiAB_max) ** h)
        elif var < phiAB_min:
            phiABx = p * ((phiAB_min - var))
        else:
            phiABx = p * ((var - phiAB_max))

        return phiABx


def barrier_function2(constraint_bounds, x, y, p, h, baseline_uwb):
    a = math.pi/2 - math.atan2(y, x)
    l = np.sqrt(x ** 2 + y ** 2)
    return barrier_function('length', constraint_bounds[0:2], l, p, h, baseline_uwb) + barrier_function('angle', constraint_bounds[2:], a, p, h, baseline_uwb)

