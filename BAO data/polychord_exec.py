# Copyright (c) Saddam Leonardo Kap 2025. If you use any of the computing program for you research using my algorithm, kindly cite my papers given in the readme file and mention the GitHub repo file in the manuscript. These codes are for purely research purpose. Do not use it for commercial purpose. You can use these algorithm and codes to teach students and give a proper credit to the originator. 
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d 
from matplotlib import rc
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from getdist import plots, MCSamples
import arviz as az

import Bao
from datetime import datetime

import new_desi_bao

import BAO_LCDM_equation  #File name, where you have put the equations, change accordingly. 
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

from scipy.stats import norm

import compress_planck

try:
    from mpi4py import MPI

except ImportError:
    pass

import new_desi_bao

# To run the program in the terminal use the following command:  mpirun -np 11 python -u polychord_exec.py

##############################
data=np.loadtxt("data/pantheon_data_M.txt")
z_data_sn=data[:,0]

data_H = np.loadtxt("cc_data.txt")
z_dataH = data_H[:, 0]



data_pl = compress_planck.data_cmb

rank = MPI.COMM_WORLD.Get_rank()

print("The rank of the cpu is ", rank)


file_name = 'lcdm_poly_cc+pla+bao'
label_fig = "CC+PLANCK"
N= len(z_dataH) + len(new_desi_bao.z_desi_bao_eff) +  len(data_pl)


def likelihood(theta):
    """
    PolyChord likelihood function.
    theta: array of parameters [om0, H0, w0, rd]
    Returns: (log_likelihood, [derived_parameters])
    """
    od0, H0, rd, rs_val, obh, Neff =  theta
    params = [od0, H0, rd, rs_val, obh, Neff]
    
    
    res = BAO_LCDM_equation.log_prob(params)
    return res, []  # No derived parameters in this case

def prior(hypercube):
    """
    PolyChord prior function.
    Transforms unit hypercube to physical parameter space.
    """
    theta = np.zeros(len(hypercube))
    # Define uniform priors matching your bounds
    theta[0] = UniformPrior(0.1, 1)(hypercube[0])  # od0
    theta[1] = UniformPrior(30.0, 100.0)(hypercube[1])  # H0
    theta[2] = UniformPrior(100, 300)(hypercube[2])  # rd
    theta[3] = UniformPrior(100, 300)(hypercube[3])  #rs

    theta[4] = UniformPrior(0.0001, 0.1)(hypercube[4]) # obh

    theta[5] = UniformPrior(1.8, 4.0)(hypercube[5])  # Neff     

    return theta

def aic(log_likelihood, ndim):
    return -2 * log_likelihood + 2 * ndim

def bic(log_likelihood, ndim, ndata):
    return -2 * log_likelihood + ndim * np.log(ndata)


nderived = 0  # No derived parameters
nlive = 500  # Number of live points

name = ['Od', 'H0', 'rd', 'rs', 'Obh', 'Neff']
labels1 = [r'\Omega_{\Lambda}', r'H_0', r'r_d', r'r_s', r'\Omega_{\rm b 0}h^2', 'N_{\rm eff}']

ndim = len(name)

ncpu = cpu_count()

print("{0} CPUs not all in used.".format(ncpu))



print(f"Now we are doing {file_name} analysis with LambdaCDm:", N)

# PolyChord settings
settings = PolyChordSettings(ndim, nderived)
settings.file_root = file_name
settings.base_dir = 'chains_new/'  #create a new directory.
settings.nlive = nlive
settings.num_repeats = ndim * 5  # Recommended for robust sampling
settings.feedback = 2  # Verbosity level  (it could be 0 to 3)
settings.do_clustering = True  # Enable clustering for multimodal posteriors
settings.read_resume = False  # Start fresh run


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

# Run PolyChord



output = pypolychord.run_polychord(likelihood, ndim, nderived, settings, prior, dumper=dumper)

print(f"Here is the result for {file_name} on:", datetime.now())

print("PolyChord run completed.")


if rank == 0:

    # paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(ndim)]  #you can also use this. 
    paramnames = name
    
    output.make_paramnames_files(paramnames)

    import getdist.plots
    
    posterior = output.posterior
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(posterior, filled=True, title_limit=1, legend_labels=[f'{label_fig}'],legend_loc='upper right', contour_colors=['orange'])
    
    
    g.export(f'figure/{file_name}.pdf')

if rank==0:
            
    print("All computations are now finished.")

# mpirun -np 11 python -u polychord_exec.py  use this command to run the file in the terminal. It will use multiproccessing. 




