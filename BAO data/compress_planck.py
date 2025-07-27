# copyright Saddam Leonardo kap (aka Saddam Hussain) 

# Follow this original paper for the data and its covariance matrix. 
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

data_cmb= np.array([1.7502, 301.471, 0.02236])

data_cmb_sigma = np.array([0.0046,0.090,0.00015])

inv_cmb = np.array([[94392.3971,-1360.4913,1664517.2916], 
                    [-1360.4913,161.4349,3671.6180],
                    [1664517.2916, 3671.6180, 79719182.5162]])

# Now you have to call this file inside the BAO+LCDM_execution.py

def planck_chi(dl_val,H_val,params, rs_val, zs_val):   
    
    om0, H0, obh = params  # this is a reduced params which is different from the global params. 

    calc_obh = obh
    
    rs= rs_val  # this you can either choose to vary or calculate it. 

    zs= zs_val
    # 

    da =  dl_val(zs)/(1+zs)**2

    la=  (1+zs)* np.pi * da/rs  #accoustic length

    Rr = ((1+zs)* da* H0* (om0)**(1/2)) /2.99792458e5  #shift parameter 
    
    shift_val = Rr 

    la_val = la  
    

    eval_val = np.array([shift_val, la_val, calc_obh])    

    res9 = eval_val - data_cmb

    chi = -0.5 * (res9.T @inv_cmb @ res9)
    
    return  chi
