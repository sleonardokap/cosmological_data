import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy import integrate 
from matplotlib import rc
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count
import emcee
from tqdm import tqdm
import scipy.linalg as la
import arviz as az

data=np.loadtxt("OHD.txt")
z_data=data[:,0]
H_data=data[:,1]
H_err= data[:,2]
errsq=1/H_err**2
h_err_use= H_err**2  # this must be 1/sigma^2 other wise it will be wrong. 
H_diag=np.diag(h_err_use)

H_inv = la.inv(H_diag)


def Hmodel(z, params):
    om0, H0 = params
    Hval=H0*np.sqrt(om0*(1+z)**3 + (1-om0))
    return Hval

def chisq(D,T,err):
    diff= D-T
    '''diffT=diff.T
    chisq=diffT.dot(errsq.dot(diff))'''
    chisq=np.sum(((D-T)/err)**2)
    return chisq

'''def chisq_H(D,T,err):
    diff= D-T
    diffT = diff.T
    chisq= np.dot(diffT,np.dot(err,diff))
    return chisq
    '''


def modelparameter(params):
    om0, H0 = params
    HT = np.zeros(len(z_data))
    if not (30 < H0 < 100 and 0 < om0 <0.6):
        return -np.inf
      
    for i in range(len(z_data)):
        HT[i] = Hmodel(z_data[i], params)
    Hubblechi = chisq(H_data, HT, H_err)
    return Hubblechi
        
def H_loglike(params):
    H_log = -0.5 * modelparameter(params)
    # Ensure that log likelihood is finite
    if not np.isfinite(H_log):
        return -np.inf
    return H_log

nwalker=100
ndim=2 #this dfines the number of model parameters that required to be constrained.
niter=10000
#let us define the initial position of workers of mcmc
p0 = np.random.uniform(low=[0.00001, 30.], high=[0.6, 99], size=(nwalker, ndim))

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))
with Pool(processes=7) as pool:
    sampler=emcee.EnsembleSampler(nwalker,ndim,H_loglike,pool=pool)
    sampler.run_mcmc(p0,niter,progress=True)
    

samples=sampler.get_chain(discard=10,thin=2,flat=True)


samples1 = sampler.chain.reshape((-1, ndim))
om_mc, H0_mc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples1, [16, 50, 84],
                                                axis=0)))
#Both these gives the same result. In below it is in tex format.
print(om_mc, H0_mc)

flat_samples=sampler.get_chain(discard=10,thin=2,flat=True)

label=[r'\Omega_M', r'H_0']

with open("math_info.txt", "w") as file:
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], label[i])
        file.write(txt + "\n")


'''for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = r"\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], label[i])
    math=Math(txt)
    np.savetxt("math.pdf")'''


params = [r'$\Omega_M$', r'$H_0$']

#Plotting the traces
thin, discard = 2, 10
fig, axes = plt.subplots(ndim, figsize=(18,9), sharex=True)

#plt.figure(figsize=(18,9))

for i in range(ndim):
    ax = axes[i]
    #plt.subplot(2,ndim//2,i+1) #this is for making two rows.
    ax.plot(sampler.get_chain()[thin:,discard:,i], color='black',alpha=0.1, lw=0.15)
    ax.set_xlim(0, len(sampler.get_chain()))
    ax.set_ylabel(params[i])

axes[-1].set_xlabel("step number");
plt.tight_layout ()

fig=corner.corner(samples,labels=params,show_titles=True,color='green',Truths=[om_mc,H0_mc],smooth=2.,Levels=[0.68,0.95],plot_density=True,plot_datapoints=False)

fig.savefig("corner_plot1.pdf")
