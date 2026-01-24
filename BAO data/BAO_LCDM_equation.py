
# copyright Saddam Leonardo Kap  
#If you use any algorithm, please don't forget to mention the repo and cite my relevant papers. 
# If you want to be a contributor, you can write me an email. 
# More advanced algorithm I will publish soon. 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from scipy.integrate import solve_ivp, simpson

from numpy.linalg import inv
import scipy.linalg as la

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
import Bao

import new_desi_bao_R2

import compress_planck

planck_like = compress_planck.planck_cmb_likelihood()  # this is important step to first initilize the likelihood file. 

##############

#Hubble data

data_H = np.loadtxt("cc_data.txt")  # Here, no-covariance matrix has been used. However, in this sample, 15 data points are highly correlated. I will update the data set and corresponding covariance matrix. 
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

# This is pantheon+ with column m_B, and apply filter. z>0.01 with cephids. The total number of sample becomes 1640 or something around that. However, you can still use 1701 data points, however, your result will not match with the original paper. This is not wrong, this mismatch is there due to the systematic error found in the nearby redshift sample. 
data_sn_pan = np.loadtxt("data_pantheon_cephids.txt") # you can change the name of your file. 
z_cmb_pan = data_sn_pan[:,0]
z_hel_pan = data_sn_pan[:,1]
mb_sn_pan = data_sn_pan[:, 2]  # this is column corresponding to m_B
cov_data_pan = np.loadtxt("covmat_pan_cephids.txt")
cov_inv_sn = inv(cov_data_pan)  # make sure to use unique name, and this has to match with the likelihood def. 




# now the equations are written in the autonomous equations framework, where the derivative takes place 
# with respect to N, x'(N) = eqx, for instruction follow the paper on k-essence: https://arxiv.org/abs/2406.07179
def equation(t, variable, params):

    #These equation have been constructed by choosing dimensionless variables as o_i = \frac{\kappa^2 \rho_i}{3 H_0^2}, where i can be chosen for any component of the universe.  Here h  = H/H_0, so in this regard the the First Friedmann equation becomes: h^2 = od + om + ob + ogam + onu; where od represents the dark energy density, om is dark matter density, ob is baryonic matter density, ogam is photon density, and onu is extrarelativistic density known as neutrino density. Similarly, from this you can construct the \dot{H}/H^2 where this is known as the 2nd Friedmann equation. The differential equation has been constructed by taking the time derivative of d(o_i)/dt, and ultimately we replace time dt by H dt which is known as dN. So, all the solution of the equation has been obtained for N, not for scale factor. The efold number N is linked with scale factor as N = ln(a) = -ln(1+z). Similarly, the differential equation for the luminosity distance dl, is also expressed in terms of N, hence, its form has been changed. Mind it here, we are obtaining the solution dl*H_0, not dl exactly, once we get back the solution, we need to divide it by H_0. Similarly, we will get the soluton h, hence to get the solution of H, we need to multiply the array of h with H_0. Here, we have adopted the differential system in order to just show the generality. In this way you can fit any parameteric model of dark energy just by replacing its equation of state and expressing it in terms of N. 

    od, om, ob, ogam, neu, h, dl = variable

    wd = -1

    # w0, w1 = params[0], params[1]

    #wd = w0 + wa* (1 - np.Exp(t))   # this is cpl equation of state with two free parameters. you can just comment out wd, and make this one active. your equation will be ready to use. Hence, by following this set up you can directly test any number of models with this algorithm. 

    doth = -3/2 * (1/h**2 * (1/3* ogam + 1/3 * neu + wd * od) + 1)

    qq = (-1-doth)

    # Make sure to change the neutrino interaction name qn1. 

    eqd =  - 3* (1+ wd) * od

    eqm = - 3 * om  
    
    eqb = -3 * ob 

    eqgam = -4 * ogam  # this is photon density. 

    eqneu = -4 * neu   # this is extra degrees of freedom. 
    
    eqh = h * doth    # this h with h = H/H0
    
    eqdl = - dl - (1 / h) * 2.99792458e5 * np.exp(- 2* t)   # this is dlH0, to get dl, one need to divide by H0. 

    var = np.array([eqd, eqm, eqb, eqgam, eqneu, eqh, eqdl])

    # if np.any(np.isnan(var)) or np.any(np.isinf(var)):
    #     return np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])

    return var



def ode_sol(params):

    # od0, H0, rd_val, obh, Neff = params
    
    od0, H0, obh= params

    Neff = 3.046  

    # rs_val = 144.39 # to obtain the valid constraint from the planck likelihood, you can either fix your rs_val or evaluate it from the formula. Below I have presented the code to estimate the value of it by doing the integration. This method is valid for any cosmological model and have not use any kind of approximations. Hence, the algorithm is highly optimized and fast. Whenever, you construct your own model, try to compute the rs_val from the formula. Never use the planck result, otherwise you will never going to see the actual change that occurs with your model. 

    # For rd_val, you can treat it as a free parameter. This parameter is corresponding to BAO. You can able to see the different values of it if you use different models with different data set. I will explain these in detail in my youtube video. 

    

    # you may fix Neff= 3.046 which is good if you only use cmb_data, since this data can't constraint N_eff, hence your H_0 won't be constrained. I will provide more generalize data later where you can get constraint on N_eff. So, my advice will be if you are not using desi Bao or BBN data you may fix Neff, other wise it won't produce good result. Once you will fix it you will get H0 = 66-68 km/s/Mpc. Once you fix Neff, kindly update your polychord or emcee sample accordingly. 

    hh=(H0/100)

    # Neff = 3.046  # for lambdaCDM.
    tt= 2.7255 # cmb temperature in kelvin.
    ogh = 3* (tt/2.7)**4/(4*31500) # this is the photon density. \Omega_{\gamma 0} h^2

    # ogh = 2.47298 * 1e-5  # this is a photon density 

    og0 = ogh/hh**2  # photon minimally coupled.

    orh = ogh* (0+ 0.227107 * Neff)  # neutrino when treated different from photon which is also a relativistic component known as the radiation. 

    ob0 = obh/hh**2
    
    or0 = orh/hh**2   # this is the neutrino density at the present time. Here or0 = \kappa^2 \rho_{\nu 0}/(3 H_0^2)

    om0 = 1- od0 - ob0- or0 - og0   # total dark matter density  # here I am treating dark matter as a different component than the baryon. 
    # this is useful when you put some interaction between the dark matter and dark energy while treating baryon matter as a standard model particle. 

    omh= om0 * hh**2  #(dark matter density. \Omega_{dm 0}h^2)

    dmh = omh+obh  # \Omega_{total matter 0}*h^2 

    w0 = 0
    wa = 0

    param = np.array([w0, wa])

    cmb_params = np.array([om0+ob0, H0, obh, omh])  # om0 refers to dark matter density. and ob0 referes to baryon matter density. obh is \Omega_{b0}(H0/100)^2

    if omh<0:
        return -np.inf
    
    if obh<0:
        return -np.inf
  

    
    tmax = 9000

    yi = np.array([od0, om0, ob0, og0, or0, 1, 0])  # this is initialization of the initial condition for all the differential equation. 


    sol = solve_ivp(lambda t, y: equation(t, y, param),
            [0, -20], yi, t_eval=np.linspace(0.0,-20,tmax), method='BDF', rtol = 1e-6, atol = 1e-3)


    t_sol = sol.t

    zz = np.exp(-t_sol)- 1     # Here, I am constructing the redshift from N using N= - ln(1+z). This is one of the greatest way to handle things efficiently. It is superior technique than solving the differential equation with respect to redshift. It boosts the speed 10x then redshift. 

    odsol, omsol, obsol, ogsol, orsol, h_sol, dlh0_sol = sol.y

    # if np.any(dlh0_sol<0):
    #     return -np.inf


    hhsol = H0 * h_sol  # This is in km/s/Mpc unit.
    dlsol =  dlh0_sol/H0  # This is in km/s/Mpc unit.  # this is normal D_L.
    

    H_val = CubicSpline(zz, hhsol)

    dl_val=CubicSpline(zz, dlsol)

    res_hubble = np.zeros(len(z_dataH))

    res_hubble = H_val(z_dataH) -  H_obs_data

    chi_hubble = -0.5 * (res_hubble.T @ H_inv @ res_hubble)


   
# Here, we are prescribing how to compute the sound horizon. This is my algorithm highly optimized and fast. If you use any of my alogirithm, don't forget to mention it and cite my article. 
    def sound_horizon_cal(z_lower):

        t_vals = t_sol
        
        
        H = h_sol  # Here we are getting h. this is varying with N. and hence we will get rd*H0 

        valid = (H>0)

        if not np.any(valid):
            return np.nan  # integral not possible
        
        if not len(valid) ==tmax:
            return np.nan

        t_vals = t_vals[valid]
        H = H[valid]

        idx = np.argsort(t_vals)
        t_vals = t_vals[idx]
        H = H[idx]

        
        H_spline = PchipInterpolator(t_vals, H)  # here it is interpolating the h values. So, that you can integrate it between any redshift, provided that your limit exists between the h evolution you have obtained from the differential equation. 

        
        def integrand_N(N):

            HN = H_spline(N)
            if HN <= 0:
                raise ValueError("Non-positive H encountered")

            arg = 3 * (1e5) * np.exp(-N) / (
                HN * np.sqrt(3 * (1 + (3 * obh) / (4 * ogh * np.exp(-N))))
            )

            # this integrand is not same as the standard intagrand. Here, I have transformed it and expressed interms of N = log a. 
            
            return arg

        # Define integration bounds (must be within t_vals range!)
        t_lower = np.log(1 / (1 + z_lower))  # in the case of rd, z_lower = 1059.94
        t_upper = -18  

        r_d, err = quad(integrand_N, t_upper, t_lower, epsabs=1e-10, epsrel=1e-10,limit=200)

        # 
        if np.iscomplex(r_d) or np.isnan(r_d):
            
            return np.nan

        return r_d

    b1= 0.313*(dmh)**(-0.419)*(1+0.607*(dmh)**0.674)

    b2 = 0.238*(dmh)**0.223

    zdd = 1345.0*((dmh)**(0.251) * (1+b1 * obh**(b2))/(1+0.659*(dmh)**0.828))

    # zdd= 1059.60

    rd_val1 = sound_horizon_cal(zdd)   # this guy calculates the rd*H0. If you wish to vary your rd as a free parameter for the BAO result, you can just comment it out, and put rd as free parameter in the initial line.  

    

    rd_val = rd_val1/H0   # this is actual rd value. 

      

    # print("The value of r_d is ", rd_val)

    if np.isnan(rd_val) or np.isinf(rd_val):
        return -np.inf
    
    g1= 0.0783*(obh)**(-0.238)/ (1+39.5*(obh)**0.763)

    g2 = 0.56*(1+21.1*(obh)**1.81)
    
    zs = 1047*(1+0.00124*(obh)**(-0.738)) * (1+g1 * (dmh)**(g2))
    # zs = 1090.0

    rs_val1 = sound_horizon_cal(zs) 

    rs_val = rs_val1/H0  # this is the sound horizon distance in Mpc at the photon recombination epoch. This estimation is necessary if you are using compress planck data. If you treat rs_val as a free paramter, you will never be going to get a good result. However, you will get a result when you will use an external data set. But it is useful not to vary rs as long as you're using planck data. However, when you use BAO data, you can vary rd as a free paramter. Since, planck data constraint your H_0, hence, it will automatically constraint the rd for BAO data. Without using BAO, you will not able to get any constraint on rd. 
        

    
    def chi_sn(dl_val, redshift):

        # Here, I am not analytically marginalizing the Mb, hence one need to use advanced likelihood to do that. This likelihood is good for only SH0ES obtained result, but if you are using m_B, then this will not give you the correct result. 

        mu_values_sn = np.zeros(len(redshift))

        res_sn = np.zeros(len(redshift))

        mu_values_sn = 5*np.log10(dl_val(redshift)) + 25

        res_sn = mu_sn -mu_values_sn

        res = -0.5 * (res_sn.T @ inverse_covar @ res_sn)

        return res  

    def chi_sn_mb_pantheon(dl_val, redshift, redshift_hel):

        # this is likelihood you should use for the Pantheon+ data when you use the column mb, and this redshfit_hel is the correction that goes with it. redshift is z_cmb, where redshift_hel is z_HEL. Don't change anything here. You have to import the data wisely. This liklihood automatically marginalizes M_B, although you feed the column m_B. So, when you plot the contour, you will not get the column corresponding to M_B. Using this likelihood method, you will always get good result aligned with pantheon plus paper. Make sure to cite my paper: https://arxiv.org/abs/2509.09202, and reference therein. 

        mu_values_sn = 5 * np.log10((1+redshift)*(1+redshift_hel) *
                                    dl_val(redshift)/(1+redshift)**2) + 25

        res_sn = mb_sn_pan - mu_values_sn
        one_vec = np.ones(len(mb_sn_pan))

        aa = res_sn @ cov_inv_sn @ res_sn
        bb = one_vec @ cov_inv_sn @ res_sn
        ccc = one_vec @ cov_inv_sn @ one_vec

        chi_sq = aa - (bb**2) / ccc
        return -0.5 * chi_sq


    # BAO calculation 

    # chi_bao_cmb = Bao.cmb_bao(dl_val, H_val, rd_val)

    # chi_sn_pantheon = chi_sn_mb_pantheon(dl_val, z_cmb_pan, z_hel_pan)

    # chi_bao_desi = new_desi_bao_R2.desi_bao(dl_val, H_val, rd_val)

    chi_planck = planck_like.planck_chi(dl_val, H_val, cmb_params, rs_val, zs)  # New likelihood has been updated for this. Don't put the value of zs by your hand. There is a code above which will compute zs for any model. 

    # you can either compute zs using the paper mentioned in the planck likelihood
    

    chi_tot =  chi_planck  # you can just do the algebraic sum of the likelihood for different data. 

    if np.any(np.isinf(chi_tot)):
        return -np.inf
        
    return chi_tot



def log_prob( params):   
    
    
    res = ode_sol(params)

    if np.any(np.isinf(res)) and np.any(np.isnan(res)):

        return -np.inf
    
    
    
    return  res


# To sample you can use emcee or polychord. Both sampling can be found in the folder. However make sure to see the parameter list. You may need to make some changes into it. 






