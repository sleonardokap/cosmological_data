
# copyright Saddam Leonardo Kap 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

import numpy as np


from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp


import scipy.linalg as la

import Bao

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




def equation( z, variable, params):
    
    
    
    od, H, dl = variable[0], variable[1], variable[2]

    om0, H0 = params[0], params[1]

    dotH = (-3 *( wd(z, params) * od + 1))
    
    eqd = 1 / (1 + z) * (3 * od *(1 + (-1)) + dotH * od)
    eqH = (3 / (2 * (1 + z))) * H * (-1 * od + 1)

    eqdl = 1/(1+z)*dl + (1/H ) * 2.99792458e5 * (1+z)
    
    return np.array([eqd, eqH, eqdl])



def ode_sol( params):

    om0 = params[0]
    H0 = params[1]
    rd = params[2]

    od0 = 1- om0
  

    
    tmax = 1000


    sol = solve_ivp(lambda t, y:equation(t,y,params), [0, 3], [od0, H0, 0], t_eval=np.linspace(0, 3, tmax), rtol=1e-2, atol=1e-2, method='RK45')


    t_sol = sol.t
   
    H_sol = sol.y[1]
    dl_sol = sol.y[2]
    
    

    H_val = interp1d(t_sol, H_sol, kind='linear', fill_value="extrapolate")

    dl_val=interp1d(t_sol, dl_sol, kind='linear', fill_value="extrapolate")

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

    chi_bao_cmb = Bao.cmb_bao(dl_val, H_val, params=params[2])

    chi_bao_desi = Bao.desi_bao(dl_val, H_val, params=params[2])
    

    chi_tot =  chi_hubble + chi_bao_desi + chi_bao_cmb 
        
    return chi_tot



def log_prob( params):
    
    
    
    res = ode_sol(params)

    if np.any(np.isinf(res)):

        return -np.inf
    
    
    
    return  res

