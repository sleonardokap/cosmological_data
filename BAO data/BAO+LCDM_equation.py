
# copyright Saddam Leonardo Kap  
#If you use any algorithm, please don't forget to mention the repo and cite my relevant papers. 
# If you want to be a contributor, you can write me an email. 
# More advanced algorithm I will publish soon. 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp


import scipy.linalg as la

import Bao

import new_desi_bao_R2

import compress_planck

##############

#Hubble data

data_H = np.loadtxt("cc_data.txt")
z_dataH = data_H[:, 0]
H_obs_data = data_H[:,1]
H_err = data_H[:, 2]
h_err_use= H_err**2  # this must be 1/sigma^2 other wise it will be wrong. 
H_diag=np.diag(h_err_use)
H_inv = la.inv(H_diag)





# Pantheon data for mu
# in this data set I am not using any filter and applied no corrections. to know more about filter and corrections regarding it, follow my paper: https://arxiv.org/pdf/2506.11755 , here I have given the link how to apply the redshift correction while computing the distance modulus.  
data_sn=np.loadtxt("pantheon_data_M.txt")
z_data_sn=data_sn[:,0]
mu_sn=data_sn[:,1]
cov_data=np.loadtxt("Pantheon_cov_all.cov")
cov_mat = cov_data.reshape(1701,1701)
inverse_covar = la.inv(cov_mat)




##########################


def wd(z,  params):
    eos = -1
    return eos



# now the equations are written in the autonomous equations framework, where the derivative takes place 
# with respect to N, x'(N) = eqx, for instruction follow the paper on k-essence: https://arxiv.org/abs/2406.07179
def equation(t, variable, params):

    od, omm, H, dl = variable

    om0, H0 = params

    orr = 1-  od - omm
        
    wdd = -1

    dotH = (-3/2 *( 1/3 * orr  +   wdd * od + 1 ))

       
        
    eqd = -1 * (3 * od *(1 + wdd) + 2 * dotH * od)

        

    eqm = - 3 * omm - 2 * omm * dotH 
        
    eqH =  H * dotH
        
    eqdl = - dl - (1 / H) * 2.99792458e5 * np.exp(- 2* t)
    
    return np.array([eqd, eqm, eqH, eqdl])



def ode_sol(params):
    om0, H0, rd, rs_val, orr0, obh = params

   

    od0 = 1- om0-orr0

    param = np.array([om0, H0])

    cmb_params = np.array([om0, H0, obh])
  

    
    tmax = 9000


    sol = solve_ivp(lambda t, y:equation(t,y,param), [0, -15], [od0, om0, H0, 0], t_eval=np.linspace(0, -15, tmax), rtol=1e-3, atol=1e-2, method='BDF')


    t_sol = sol.t
   
    odsol, omsol, H_sol, dl_sol = sol.y

    zsol = np.exp(-t_sol)- 1  # Here, I am constructing the redshift from N using N= - ln(1+z). This is one of the greatest way to handle things efficiently. It is superior technique than solving the differential equation with respect to redshift. It boosts the speed 10x then redshift. 
        
    

    H_val = interp1d(zsol, H_sol, kind='cubic')

    dl_val=interp1d(zsol, dl_sol, kind='cubic')

    res_hubble = np.zeros(len(z_dataH))

    res_hubble = H_val(z_dataH) -  H_obs_data

    chi_hubble = -0.5 * (res_hubble.T @ H_inv @ res_hubble)

    
    def chi_sn(dl_val, redshift):

        mu_values_sn = np.zeros(len(redshift))

        res_sn = np.zeros(len(redshift))

        mu_values_sn = 5*np.log10(dl_val(redshift)) + 25

        res_sn = mu_sn -mu_values_sn

        res = -0.5 * (res_sn.T @ inverse_covar @ res_sn)

        return res    


    # BAO calculation 

    chi_bao_cmb = Bao.cmb_bao(dl_val, H_val, params=rd)

    chi_bao_desi = new_desi_bao_R2.desi_bao(dl_val, H_val, params=rd)

    chi_planck = compress_planck.planck_chi(dl_val, H_val, cmb_params, rs_val, 1089.90)

    # you can either compute zs using the paper mentioned in the planck likelihood
    

    chi_tot =  chi_hubble + chi_bao_desi + chi_bao_cmb + chi_planck
        
    return chi_tot



def log_prob( params):   
    
    
    res = ode_sol(params)

    if np.any(np.isinf(res)) and np.any(np.isnan(res)):

        return -np.inf
    
    
    
    return  res






