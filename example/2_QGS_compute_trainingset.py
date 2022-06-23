"""
Python script to compute a training set for a machine learning method. If the training is computed from observation, a DA algorithm is applied.
run 'python compute_trainingset.py -h' to see the usage
"""

# In[QGS HMM setting]
import numpy as np
from dapper.admin import HiddenMarkovModel

import time
from multiprocessing import freeze_support, get_start_method
from dapper import *
from dapper.tools.convenience import simulate

from dapper.mods.Qgs.qgs.params.params import QgParams
from dapper.mods.Qgs.qgs.functions.tendencies import create_tendencies
from dapper.mods.Qgs.qgs.integrators.integrator import RungeKuttaIntegrator


if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()
        
    a_nx = 2 
    a_ny = 2 
    o_nx = 2 
    o_ny = 4 
    Na = a_ny*(2*a_nx+1) 
    No = o_ny*o_nx


    Nx = 2*Na+2*No
    #start Time
    T1 =time.perf_counter()
    ################
    # General parameters
    # Setting some model parameters
    # Model parameters instantiation with default specs
    model_parameters = QgParams()
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(a_nx, a_ny)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(o_nx, o_ny)

# Setting MAOOAM parameters according to the publication linked above
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                                 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06, })
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

    model_parameters.print_params()

    f, Df = create_tendencies(model_parameters)   
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    
    # Saving the model state n steps
    def step(x0, t0, dt):
       y = x0
       integrator.integrate(t0, t0+dt, 0.01, ic=y, write_steps=0)
       t0, y0 = integrator.get_trajectories()
       #t,y = integrate_runge_kutta(f, t0, t0+dt, 0.1, ic=y, forward=True,write_steps=0, b=None, c=None, a=None)
       return y0

    integration_time = 4.e5
    dt = 10.0
    dto= 10.0
    t = Chronology(dt=dt, dtObs=dto, T=integration_time,  BurnIn=0)
    sig_hf = np.load('sigma_hf.npy')
#parameter setting
    #1:Dyn
    sig_m = np.copy(sig_hf)
    sig_m = sig_m*0.001
    sig_o = 0.1*np.copy(sig_hf)
    sig_m[20:36]=0.0
    sig_m = sig_m*0.001
    Dyn = {
        'M': Nx,
        'model': step,
        'linear': Df,
        'noise': sig_m,
    }
    #2:Time

    #3:X0
    from utils import sampling
    X0 = RV(36, func=sampling)

    jj = np.arange(Nx)  # obs_inds
    Obs = partial_direct_Obs(Nx, jj)
    Obs['noise'] = GaussRV(C=sig_o.T*np.eye(Nx))
    HMM = HiddenMarkovModel(Dyn, Obs, t, X0)

# In[save data]
   #use the DA result to train the CNN
    data =np.load('daPF.npz')
    xx = data['mua']
    #Select training time series
    xx_train = xx[0:30000]

	# Estimation of the true value after dtObs MTU
    xx_out = xx_train[1:]
    xx_in = xx_train[:-1]
    
    dtObs = 10.0
	# Truncated value after dtObs MTU
    Dyn = HMM.Dyn
    xx_trunc = step(xx_in, 0, dtObs)
    delta = (xx_out - xx_trunc) / dtObs
    xxm = np.mean(xx_in,axis=0)
    xxstd =np.std(xx_in,axis=0)
    xx_in_n = (xx_in-xxm)/xxstd


#save the training file
    np.savez('train_QGS.npz', x=xx_in, y=delta, dtObs=dtObs)
    np.savez('train_QGS_normalize1.npz', x=xx_in_n, y=delta, dtObs=dtObs)
    np.save('xmean.npy', xxm)
    np.save('xstd.npy', xxstd)

