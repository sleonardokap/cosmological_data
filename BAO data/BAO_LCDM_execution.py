
# copyright Saddam Leonardo Kap 
#cite the relevant papers and mentioned the repo in your work if you use my algorithm 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

# IF you are using any algorithm or piece of code, kindly mention my code and cite my articles: https://arxiv.org/abs/2509.09202 and reference therein. You can also find many references of mine in the Readme.md file. Let me know if you find any difficulty or error. 

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d 
from matplotlib import rc
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import getdist
from getdist import plots, MCSamples

import arviz as az

import emcee
import BAO_LCDM_equation
import Bao

from datetime import datetime

import compress_planck
import new_desi_bao_R2



data=np.loadtxt("pantheon_data_M.txt")
z_data_sn=data[:,0]

data_H = np.loadtxt("cc_data.txt")
z_dataH = data_H[:, 0]


data_pl = compress_planck.data_cmb # planck data. 

##############################



N= len(z_dataH) + len(new_desi_bao_R2.desz) + len(Bao.cmbbao_z) + len(data_pl)

file_name= "lcdm_BAO+CC+pla"
label_fig="CC+BAO+PLANCK"

# bao refers to SDSS BAO

####################


    

def aic(log_liklihood,ndim):

    res = -2 * log_liklihood + 2 * ndim
    return res

def bic(log_liklihood,ndim,ndata):
    return -2 * log_liklihood + ndim * np.log(N)



def liklihood(params):

    od0, H0, rd, rs_val, obh, Neff = params
    
    if not 0.1 < od0 < 1: 
        return -np.inf
    
    if not 30 < H0 < 100: 
        return -np.inf
    
    if not 100 < rd< 300:
        return -np. inf

    if not 100 < rs_val< 300:
        return -np. inf

            
    if not 0.00001 <obh <0.1:
        return -np.inf

    if not 1.9 <Neff<4.0:
        return -np.inf
    
    
    res = BAO_LCDM_equation.log_prob(params)
    
    return res


ncpu = cpu_count()

print("{0} CPUs not all in used.".format(ncpu))



print(f"Now we are doing {file_name} analysis with LambdaCDm:", N)

name = ['Omega_d', 'H0', 'rd', 'rs', 'Obh', 'Neff']
labels1 = [r'\Omega_{\Lambda}', r'H_0', r'r_d', r'r_s', r'\Omega_{\rm b}h^2', r'N_{\rm eff}']

nwalker = 60
ndim = len(name)
niter = 400000

p0 = np.random.uniform(low=[0.1, 40.,100, 100, 0.00001, 1.9], high=[1, 100,300, 300,0.1, 4.0], size=(nwalker, ndim))

move = [
        (emcee.moves.StretchMove(a=2.0),0.30),
        (emcee.moves.DEMove(), 0.50),
        (emcee.moves.DESnookerMove(), 0.20)
        # (emcee.moves.KDEMove(bw_method=None), 0.30)
    ]
with Pool(processes=10) as pool:  #set the processes according to the number of cores you have. 
    sampler = emcee.EnsembleSampler(nwalker, ndim, liklihood,pool=pool,live_dangerously=None,  moves=move)
    sampler.run_mcmc(p0,niter,progress=True)

dis=4000
thi=10

chains = sampler.get_chain(flat=True,discard=dis, thin=thi)



sample2 = MCSamples(samples=chains,names=name, labels=labels1)



print(f"Here is the result for {file_name} on:")

az_data = az.from_emcee(sampler, var_names=name)
r_hat = az.rhat(az_data)
print("Gelman-Rubin statistic (R_hat):", r_hat)



log_likelihoods = sampler.get_log_prob()

log_likelihood = np.max(log_likelihoods)

print("Maximum log liklihood is ", log_likelihood)

max_likelihood_index = np.argmax(log_likelihood)
max_likelihood_params = sample2[max_likelihood_index]

chi2val = -2* log_likelihood
print("Min chi_square: ", chi2val)

reduced = chi2val/(N-ndim)

print("Reduced chi_squre is: ", reduced)

print(f"Parameters at Maximum Likelihood: {max_likelihood_params}\n")




aic_value = aic(log_likelihood,ndim)
bic_value=bic(log_likelihood,ndim,N)


print("AIC values:", aic(log_likelihood,ndim))

print("BIC Values: ", bic_value)


   

g = plots.get_subplot_plotter(width_inch=6)  # Adjust width of the plot in 

g.settings.figure_legend_frame = True  # Remove frame around the legend
g.settings.alpha_filled_add = 0.6  # Adjust transparency of filled contours
g.settings.title_limit_fontsize = 14  # Limit title fontsize to 14
g.settings.axes_labelsize = 12  # Adjust label fontsize
g.settings.legend_fontsize = 10  # Adjust legend fontsize
g.settings.colorbar_axes_fontsize = 10  # Adjust colorbar fontsize
g.triangle_plot(sample2, name, filled=True, legend_labels=[f'{label_fig}'],legend_loc='upper right', contour_colors=['orange'],title_limit=1)  

g.export(f'figure/{file_name}.pdf')






