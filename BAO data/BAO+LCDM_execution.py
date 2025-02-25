
# copyright Saddam Leonardo Kap 

# Tutorial link: https://youtu.be/iRYb6wQVaO8

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
import new_lcdm
import Bao

from datetime import datetime



data=np.loadtxt("pantheon_data_M.txt")
z_data_sn=data[:,0]

data_H = np.loadtxt("cc_data.txt")
z_dataH = data_H[:, 0]




##############################



N= len(z_dataH) + len(Bao.desz) + len(Bao.cmbbao_z) 

file_name= "lcdm_BAO+CC"
label_fig="CC+BAO"

# bao refers to SDSS BAO

####################


    

def aic(log_liklihood,ndim):

    res = -2 * log_liklihood + 2 * ndim
    return res

def bic(log_liklihood,ndim,ndata):
    return -2 * log_liklihood + ndim * np.log(N)



def liklihood(params):

    om0 = params[0]
    H0 = params[1]
    rd = params[2]
    if not 0 < om0 < 0.7: 
        return -np.inf
    
    if not 30 < H0 < 100: 
        return -np.inf
    
    if not 100 < rd< 300:
        return -np. inf
    
    
    res =new_lcdm.log_prob(params)
    
    return res


ncpu = cpu_count()

print("{0} CPUs not all in used.".format(ncpu))



print(f"Now we are doing {file_name} analysis with LambdaCDm:", N)


nwalker = 30
ndim = 3
niter = 20000

p0 = np.random.uniform(low=[0.0, 40.,100], high=[0.7, 100,300], size=(nwalker, ndim))

with Pool(processes=10) as pool:
    sampler = emcee.EnsembleSampler(nwalker, ndim, liklihood,pool=pool,live_dangerously=None)
    sampler.run_mcmc(p0,niter,progress=True)

dis=1500
thi=10

chains = sampler.get_chain(flat=True,discard=dis, thin=thi)

name = ['Omega_m', 'H0', 'rd']
labels1 = [r'\Omega_{m}', r'H_0', r'r_d']

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


