import math
import numpy as np

def adaptive_constraint_bound(constraint_bounds, T, alpha, beta, baseline_uwb, phat_AB):

    # phat AB alsready deocmposed into cos and sine components 
    # baseline UWB contains the scalar length and the angle ...

    lAB_min = constraint_bounds[0] 
    lAB_max = constraint_bounds[1] 
    phiAB_min = constraint_bounds[2] 
    phiAB_max = constraint_bounds[-1] 

    ptilda_AB = np.zeros(2)
    x = baseline_uwb[0] * math.sin(baseline_uwb[1]) 	# angle in radians hopefully
    y = baseline_uwb[0] * math.cos(baseline_uwb[1])
    ptilda_AB[0] = x
    ptilda_AB[1] = y

    omega = compute_omega(ptilda_AB, phat_AB)

    new_constraint_bound = np.zeros_like(constraint_bounds)

    theo_max = np.sqrt(lAB_min**2 + (phiAB_min * baseline_uwb[0])**2)

    if omega/theo_max <= T:
        lAB_min /= alpha 
        lAB_max /= alpha 
        phiAB_min /= alpha 
        phiAB_max /= alpha 
    else:
        lAB_min *= beta 
        lAB_max *= beta 
        phiAB_min *= beta 
        phiAB_max *= beta 

    new_constraint_bound[0] = lAB_min
    new_constraint_bound[1] = lAB_max
    new_constraint_bound[2] = phiAB_min
    new_constraint_bound[3] = phiAB_max
    return new_constraint_bound



def compute_omega (baseline_uwb, phat_AB):
    # for starters use the 2 norm difference
    term = baseline_uwb - phat_AB
    return np.linalg.norm(term)