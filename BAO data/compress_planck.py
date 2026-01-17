# copyright Saddam Leonardo kap (aka Saddam Hussain) 

# Follow this original paper for the data and its covariance matrix. 
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# This data set has been taken from new paper:
# https://www.aanda.org/articles/aa/pdf/2020/07/aa36720-19.pdf  
# to understand the likelihood please follow this paper, and if you are using my code or any algorithm,
# please try to mention it in the paper and cite my work mentioned in the Readme.md. 

data_cmb= np.array([2.237,1.0411,1.74998]) 

data_cmb_sigma = np.array([0.015,0.00031,0.004])

inv_cmb = np.array([[ 7.40859769e+03, -2.28256250e+04,  1.66890785e+04],
 [-2.28256250e+04,  1.32689897e+07,  4.19113942e+05],
 [ 1.66890785e+04,  4.19113942e+05,  1.16869360e+05]])

# Now you have to call this file inside the BAO+LCDM_execution.py


class planck_cmb_likelihood:

  def planck_chi(self, dl_val,H_val, params, rs_val, zs_val):  

    om0, H0, obh = params  # this is a reduced params which is different from the global params. 

    calc_obh = 100* obh
    
    rs= rs_val  # this you can either choose to vary or calculate it. 

    zs= zs_val
    # 

    da =  dl_val(zs)/(1+zs)

    theta = rs/da

    Rr = (da* H0* (om0)**(1/2))/2.99792458e5  #shift parameter 
    
    shift_val = Rr 

    theta_val = 100* theta
    

    eval_val = np.array([calc_obh, theta_val, shift_val])  

    res9 = eval_val - data_cmb

    chi = -0.5 * (res9.T @inv_cmb @ res9)
    
    return  chi
