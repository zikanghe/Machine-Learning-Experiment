"""
Created on Fri Apr  8 10:17:03 2022
This code is used to configure Configure the HMM and the collection members needed to generate the  initial ensemble field

@author: Zikang He
"""

import numpy as np

import dapper.mods as modelling
import time
from multiprocessing import freeze_support, get_start_method
# Importing the model's modules

# Start on a random initial condition

from dapper.mods.Qgs.qgs.params.params import QgParams
from dapper.mods.Qgs.qgs.functions.tendencies import create_tendencies
from dapper.mods.Qgs.qgs.integrators.integrator import RungeKuttaIntegrator
from dapper.mods.Qgs.qgs.integrators.integrate import integrate_runge_kutta

###############
#model defalt parameter
###############
default_prms = {        
   'a_nx'             : 2, 
   'a_ny'             : 2, 
   'o_nx'             : 2, 
   'o_ny'             : 4, 
   'transient_time'   : 3.e6,
   'integration_time' : 10000    
}
D = default_prms.copy()
Na = D['a_ny']*(2*D['a_nx']+1)
No = D['o_ny']*D['o_nx']
Nx = 2*Na+2*No
################
# General parameters
# Setting some model parameters
# Model parameters instantiation with default specs
      
model_parameters = QgParams()
# Mode truncation at the wavenumber 2 in both x and y spatial coordinate
model_parameters.set_atmospheric_channel_fourier_modes(D['a_nx'], D['a_ny'])
# Mode truncation at the wavenumber 2 in the x and at the
# wavenumber 4 in the y spatial coordinates for the ocean
model_parameters.set_oceanic_basin_fourier_modes(D['o_nx'], D['o_ny'])

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
   
#parameter setting

    #1:Dyn
Dyn = {
        'M': Nx,
        'model': step,
        'linear': Df,
        'noise': 0.0,
}


to_time = D['transient_time'] + D['integration_time']
dt = 100*0.1
    #day=1/model_parameters.dimensional_time
t = modelling.Chronology(dt=dt, dto=dt*100, T=to_time,  BurnIn=['transient_time'] )
    
np.random.seed(21217) 
ic = np.random.rand(model_parameters.ndim)*0.01
x0=ic
X0 = modelling.GaussRV(mu=x0, C=0.0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.0000002  # modelling.GaussRV(C=CovMat(2*eye(Nx)))
    
HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0)

