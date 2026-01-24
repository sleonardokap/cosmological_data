# copyright Saddam Leonardo kap (aka Saddam Hussain) 

# Follow this original paper for the data and its covariance matrix. 
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
from numpy.linalg import inv

# This data set has been taken from new paper:
# https://www.aanda.org/articles/aa/pdf/2020/07/aa36720-19.pdf  
# to understand the likelihood please follow this paper, and if you are using my code or any algorithm,
# please try to mention it in the paper and cite my work mentioned in the Readme.md. 

# data_cmb= np.array([2.237,1.0411,1.74998]) 

# data_cmb_sigma = np.array([0.015,0.00031,0.004])

# inv_cmb = np.array([[ 7.40859769e+03, -2.28256250e+04,  1.66890785e+04],
#  [-2.28256250e+04,  1.32689897e+07,  4.19113942e+05],
#  [ 1.66890785e+04,  4.19113942e+05,  1.16869360e+05]])

# We can use this data too however, we will not use it if we need to test our model behyond standard csomology. 

# cov_matrix_pla= np.array([[1.0, 0.34, -0.63],
#                    [0.34, 1.0, -0.46],
#                    [-0.63, -0.46, 1.0]])

#=========== We shall use this data set for testing model beyond lambdaCDM. =======
# Here the data structure is [100*Omega_{b0}h^2, 100 theta_* , R, Omega_{c0}h^2]

data_cmb = np.array([2.225,1.0414,1.7529,0.1184])

data_cmb_sigma_gen = np.array([0.0223,0.00054,0.0056, 0.0029])


inv_cmb_gen = np.array(
    [[ 8.36144553e+03, -1.94826450e+04,  2.42268487e+04, -2.68410558e+04],
 [-1.94826450e+04,  1.07036647e+07, -2.12229562e+05,  1.61294718e+06],
 [ 2.42268487e+04, -2.12229562e+05,  1.05554887e+05, -8.82933658e+04],
 [-2.68410558e+04 , 1.61294718e+06, -8.82933658e+04 , 4.35051678e+05]])



# Now you have to call this file inside the BAO+LCDM_execution.py


class planck_cmb_likelihood:

  def planck_chi(self, dl_val,H_val,params, rs_val, zs_val):

    om0, H0, obh, dmh = params    # Here dmh \Omega_dm h^2

    calc_obh =100* obh
    
    rs= rs_val

    

    zs = zs_val  #this can either be fixed and must be set TT+EE+lowE data for flat LambdaCDM plank 18.  

    # https://doi.org/10.1016/j.dark.2024.101641

    da = dl_val(zs)/(1+zs)**1

    # la= (1+zs) * np.pi * da/rs  #accoustic length

    theta = rs/da

    Rr = (da* H0* (om0)**(1/2))/2.99792458e5 #shift parameter


    
    shift_val = Rr 

    theta_val = 100* theta
    
    # ob_val = ob0 * h**2

    # eval_val = np.array([shift_val, la_val, ob_val])

    eval_val = np.array([calc_obh, theta_val, shift_val, dmh ])

    

    res9 = eval_val - data_cmb

    # res_cmb = solve_triangular(chol_cmb, res9, lower=True, check_finite=False)

    chi = -0.5 * (res9.T @inv_cmb_gen @ res9)
    
    return  chi
