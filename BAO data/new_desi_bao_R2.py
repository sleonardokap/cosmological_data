# this file can be used for the newly released DESI BAO DRII. Here, you can use this file to integrate with my previously relaesed executable file.  
# copyright Saddam Leonardo Kap. 
# If you are using my any form of algorithm, don't forget to mention my github repo and cite my published articles.  

import numpy as np

z_desi_bao_eff = np.array([0.295,0.510, 0.706, 0.934, 1.321, 1.484, 2.330])  # this is the redshift we are using to calculate the observables. 
dat_bao=np.loadtxt('desi_mean.txt')
cov_bao=np.loadtxt("desi_cov.txt") # this is the inverse of covariance matrix, although I named it cov_bao. 

#these files can be found here in the BAO data folder. Download and run. 

def desi_bao(dl_val, H_val, rd_val):

    rd = rd_val   # Here, you can simply vary the rd in the uniform range [100,300].

    dm_for_dv = dl_val(0.29500000)/(1+0.29500000)

    
    dv0 = (0.29500000 * dm_for_dv**2 *2.99792458e5/H_val(0.29500000) )**(0.333)/rd

    dm05 = dl_val(0.51000000)/(1+0.51000000)/rd
    dh05 = 2.99792458e5/(rd* H_val(0.51000000))

    dm07 = dl_val(0.70600000)/(1+0.70600000)/rd
    dh07 = 2.99792458e5/(rd* H_val(0.70600000))

    dm09 = dl_val(0.93400000)/(1+0.93400000)/rd
    dh09 = 2.99792458e5/(rd* H_val(0.93400000))

    dm13 = dl_val(1.32100000)/(1+1.32100000)/rd
    dh13 = 2.99792458e5/(rd* H_val(1.32100000))

    dm14 = dl_val(1.48400000)/(1+1.48400000)/rd
    dh14 = 2.99792458e5/(rd* H_val(1.48400000))

    dm23 = dl_val(2.33000000)/(1+2.33000000)/rd
    dh23 = 2.99792458e5/(rd* H_val(2.33000000))

    
    val = np.array([dv0, dm05, dh05, dm07,dh07, dm09, dh09, dm13, dh13, dm14, dh14, dh23,dh23])


    res = dat_bao - val

    chi_bao = -0.5* (res @ cov_bao @ res )    


    return chi_bao
