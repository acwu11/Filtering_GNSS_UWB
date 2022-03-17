import math
import numpy as np

def baseline_from_uwb(uwb_meas):
    '''
    takes measurements from uwb [range, angle] and returns estimated coords.
    and constraint bounds
    '''

    # all the hyperparameters here.
    fac = 10
    sigma_dist = 0.05
    sigma_angle = 1e-3


    baseline_uwb = np.zeros(2)
    baseline_uwb[0] = uwb_meas[0]
    baseline_uwb[1] = math.radians(uwb_meas[1] + uwb_meas[2])


    #stores the length and the angle
    constraint_bounds = np.zeros(4)        # lmin, lmax, phi_min, phi_max
    constraint_bounds[0] =  fac * sigma_dist
    constraint_bounds[1] =  fac * sigma_dist
    constraint_bounds[2] =  fac * sigma_angle
    constraint_bounds[3] =  fac * sigma_angle

    return baseline_uwb, constraint_bounds
