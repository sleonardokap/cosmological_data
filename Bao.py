
# copyright Saddam Leonardo Kap 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


# CMB BAO data

cmbbao_z = np.array([0.15, 0.38, 0.51, 0.70,0.85,1.48,2.33,2.33]) 

dmdhz = np.array([0.38,0.51,0.70,1.48, 2.33, 2.33])

dmdata= np.array([10.23, 13.36, 17.86, 30.69, 37.6, 37.3])
dmsigma = np.array([0.17, 0.21, 0.33, 0.80, 1.9, 1.7])

dhdata = np.array([25, 22.33, 19.33, 13.26, 8.93, 9.08])

dhsigma = np.array([0.76, 0.58, 0.53, 0.55, 0.28, 0.34])

dvz = np.array([0.15, 0.85])

dvdata = np.array([4.47, 18.33])

dvsigmaplus = np.array([0.17, 0.57])
dvsigmaminus = np.array([0.17, 0.62])



def cmb_bao(dl_val, H_val, params):

    rd = params

    dmrdval = np.zeros(len(dmdhz))
    dhrdval = np.zeros(len(dmdhz))
    dm_for_dv = np.zeros(len(dvz))
    dvdr_val = np.zeros(len(dvz))
    res_dmrd = np.zeros(len(dmdhz))
    res_dhrd = np.zeros(len(dmdhz))
    res_dvrd = np.zeros(len(dvz))


    dmrdval = dl_val(dmdhz)/(rd* (1 +dmdhz))

    dhrdval = 2.99792458e5/(rd* H_val(dmdhz))

    dm_for_dv = dl_val(dvz)/(1+dvz)

    dvdr_val = (dvz * dm_for_dv**2 *2.99792458e5/H_val(dvz) )**(1/3)/rd

##############

    res_dmrd = dmrdval - dmdata

    chi_dmrd = -0.5* np.sum((res_dmrd/dmsigma)**2)


##############


    res_dhrd = dhrdval - dhdata

    chi_dhrd = -0.5* np.sum((res_dhrd/dhsigma)**2)


##############

    res_dvrd = dvdr_val - dvdata

    chi_dvrd = -0.5* 0.5*(np.sum((res_dvrd/dvsigmaminus)**2) + np.sum((res_dvrd/dvsigmaplus)**2))


    chi_tot = chi_dmrd +  chi_dvrd + chi_dhrd

    return chi_tot


## DESI BAO data. 



desz = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330])

desi_dmdh_z = np.array([0.510,0.706,0.930,1.317,2.330])

desi_dmrd_data = np.array([13.62, 16.85,21.71,27.79, 39.71])

desi_dmrd_sigma = np.array([0.25, 0.32, 0.28, 0.69, 0.94])

desi_dhrd_data = np.array([20.98, 20.08, 17.88, 13.82, 8.52])

desi_dhrd_sigma = np.array([0.61, 0.60, 0.35, 0.42, 0.17])

desi_dvz = np.array([0.295, 1.491])

desi_dvrd_data = np.array([7.93, 26.07])

desi_dvrd_sigma = np.array([0.15, 0.67])



def desi_bao(dl_val, H_val, params):

    rd = params

    dmrdval = np.zeros(len(desi_dmdh_z))
    dhrdval = np.zeros(len(desi_dmdh_z))
    dm_for_dv = np.zeros(len(desi_dvz))
    dvdr_val = np.zeros(len(desi_dvz))
    res_dmrd = np.zeros(len(desi_dmdh_z))
    res_dhrd = np.zeros(len(desi_dmdh_z))
    res_dvrd = np.zeros(len(desi_dvz))


    dmrdval = dl_val(desi_dmdh_z)/(rd* (1 +desi_dmdh_z))

    dhrdval = 2.99792458e5/(rd* H_val(desi_dmdh_z))

    dm_for_dv = dl_val(desi_dvz)/(1+desi_dvz)

    dvdr_val = (desi_dvz * dm_for_dv**2 *2.99792458e5/H_val(desi_dvz) )**(1/3)/rd

##############

    res_dmrd = dmrdval - desi_dmrd_data

    chi_dmrd = -0.5* np.sum((res_dmrd/desi_dmrd_sigma)**2)


##############


    res_dhrd = dhrdval - desi_dhrd_data

    chi_dhrd = -0.5* np.sum((res_dhrd/desi_dhrd_sigma)**2)


##############

    res_dvrd = dvdr_val - desi_dvrd_data

    chi_dvrd = -0.5* (np.sum((res_dvrd/desi_dvrd_sigma)**2) )


    chi_tot = chi_dmrd + chi_dvrd + chi_dhrd

    return chi_tot



if __name__=="__main__":
    
    cmb_bao()
    
    desi_bao()

