# copyright Saddam Leonardo Kap 
# For Tutorial go to link: https://youtu.be/u88Goguzm9M

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import rc
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count
import emcee
from tqdm import tqdm
import getdist
from getdist import plots, MCSamples
import scipy.linalg as la

data=np.loadtxt("pantheon_data_M.txt") # Data columns from pantheon + sample
z_data_sn=data[:,0]                         # z data column from pantheon + sample
mu_sn=data[:,1]                             # \mu data column from pantheon + sample
cov_data=np.loadtxt("Pantheon_cov_all.cov") #Covariance Matrix of Pantheon+ Sample
cov_mat = cov_data.reshape(1701,1701)
inverse_covar = la.inv(cov_mat)

def wd(z, params):
    od0, H0, w0 = params
    eos = w0
    return eos

def equation(z, variable, params):
    od, H, dl = variable

    od0, H0, w0 = params
        
    dotH = (-3 *( wd(z, params) * od + 1))
    
    eqd = 1 / (1 + z) * (3 * od *(1 + wd(z,params)) + dotH * od)
    eqH = (3 / (2 * (1 + z))) * H * (wd(z, params) * od + 1)

    eqdl = 1/(1+z)*dl + (1/H ) * 2.99792458e5 * (1+z)

    return [eqd, eqH, eqdl]

def mu_model(z, params):
    
    od0, H0, w0 = params

    
    mu_values=np.zeros(len(z_data_sn))
    
    sol = solve_ivp(lambda t, y: equation(t, y, params), [0, 3], [od0, H0, 0], t_eval=np.unique(z_data_sn),method='RK45')
        
    
    dl_sol =sol.y[2]

    dl_val=interp1d(sol.t, dl_sol, kind='linear', fill_value="extrapolate")
    
    mu_val= 5*np.log10(dl_val(z)) + 25 
    
    return mu_val

   

def chisq(D,T,err):
    diff= D-T
    diffT = diff.T
    chisq= np.dot(diffT,np.dot(err,diff))
    return chisq

def log_prior(params):
    od0, H0, w0 = params
    if  not (40 < H0 < 99):
        return -np.inf
    if  not (0.5 < od0 < 1):
        return -np.inf
    if not (-1.3 < w0 < -0.3):
        return -np.inf
    
    return 0

def log_prob(params):
    prior = log_prior(params)
    if prior == -np.inf:
        return -np.inf
    
    # If prior is valid, proceed with log-likelihood calculation
    mu = mu_model(z_data_sn, params)
    if np.any(np.isinf(mu)):  # Check if mu_model returned -np.inf
        return -np.inf


    chisq_value = chisq(mu_sn, mu, inverse_covar)
    return -0.5 * chisq_value 



def aic(log_liklihood,ndim):
    return -2 * log_liklihood + 2 * ndim

def bic(log_liklihood,ndim,ndata):
    return -2 * log_liklihood + ndim * np.log(ndata)


nwalker = 30
ndim = 3  # Number of model parameters
niter = 50000

p0 = np.random.uniform(low=[0.5, 40., -0.8], high=[1, 99, -1.2], size=(nwalker, ndim))

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))


with Pool(processes=10) as pool:
    sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob,pool=pool)
    sampler.run_mcmc(p0,niter,progress=True)

# Define the initial position of walkers for MCMC

dis=1000
th=20

chains = sampler.get_chain(flat=True,discard=dis, thin=th)


samples=sampler.get_chain(discard=dis,thin=th,flat=True)

name = ['od0', 'H0', 'w0'] # This is must to generate the MCSamples.

labels1 = [r'\Omega_d', r'H_0', r'w_0']  # Labels for the plot

sample2 = MCSamples(samples=samples,names=name, labels=labels1)  

log_likelihoods = sampler.get_log_prob()
log_likelihood = np.mean(log_likelihoods)


print("log liklihood is ", log_likelihood)

# print("mu values:", lcdm_only_de_sn.mu_model(z_data_sn,p0))
print("AIC values:", aic(log_likelihood,4))



aic_value = aic(log_likelihood,ndim)
bic_value=bic(log_likelihood,ndim,len(z_data_sn))


# Convert MCMC chains to GetDist format
names = ['od0', 'H0', 'w0']  # Names of the parameters
labels = [r'\Omega_d', r'H_0', r'w_0']  # Labels for the plot

sample2 = MCSamples(samples=chains, names=names, labels=labels)

#samples = sampler.get_chain(discard=5, thin=1, flat=True)

samples1 = sampler.chain.reshape((-1, ndim))
od_mc, H0_mc, w0_mc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples1, [16, 50, 84],
                                                axis=0)))
#Both these gives the same result. In below it is in tex format.
print(od_mc, H0_mc, w0_mc)



flat_samples=sampler.get_chain(discard=10,thin=2,flat=True)

label=[r'\Omega_d', r'H_0', r'w_0']

with open("SN_statstics_basic.txt", "w") as file:
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], label[i])
        file.write(txt + "\n")
    file.write(f"AIC value = {aic_value:.2f}\n")
    file.write(f"BIC value = {bic_value:.2f}\n")


g = plots.get_subplot_plotter(width_inch=6)  # Adjust width of the plot in inches
g.settings.figure_legend_frame = True  # Remove frame around the legend
g.settings.alpha_filled_add = 0.6  # Adjust transparency of filled contours
g.settings.title_limit_fontsize = 14  # Limit title fontsize to 14
g.settings.axes_labelsize = 12  # Adjust label fontsize
g.settings.legend_fontsize = 10  # Adjust legend fontsize
g.settings.colorbar_axes_fontsize = 10  # Adjust colorbar fontsize
g.triangle_plot(sample2,['od0', 'H0', 'w0'], filled=True, contour_colors=['red'],title_limit=1)  # Set contour colors as desired


g.export('SN_chain_git.pdf')


# The triangular plot has been updated in the main section. Follow the file name "SN_chain_git.pdf" 
