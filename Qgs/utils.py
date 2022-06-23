import numpy as np
import dapper.mods as modelling



def sampling(N):
 """
:param N: size of the ensemble to sample
:return: the ensemble
 """
 data = np.load('true.npy')  
 N0 = data.shape[0]
 save_state = np.random.get_state()
 np.random.seed()
 idx = np.random.choice(N0, N, replace=True)
 np.random.set_state(save_state)
 E = data[idx]
 return E

def sampling2(N2):

  data = np.load('ten.npy')  
  N0 = data.shape[0]
  save_state = np.random.get_state()
  np.random.seed(1)
  idx = np.random.choice(N0,N2,replace=True)
  np.random.set_state(save_state)
  E = data[idx]
  return E
  
def mean_fileter(x1,N):
    v=0
    beta1=1-1/N
    x = np.copy(x1)
    for i in range(x.shape[0]):
        x[i]=(beta1*v+(1-beta1)*x1[i])
        v=x[i]
    return x

def compute_tinds( self ):
		"""
		compute the index of observation for each time steps
		"""
		self.tinds = dict()
		save_state = np.random.get_state()
		np.random.seed(self.seed_obs)
		for k, KObs, t_, dt in self.t.forecast_range:
			if KObs is not None:
				if self.sample == 'random':
					self.tinds[t_] = np.random.choice(self.m, size=self.p, replace=False)
				elif self.sample == 'regular':
					self.tinds[t_] = np.linspace(0, self.m, self.p, endpoint=False, dtype=np.int)
		np.random.set_state(save_state)
def def_hmod( self ):
		"""
		:return: the observation operator
		"""
		@ens_compatible
		def hmod ( E, t ):
			return E[tinds[t]]
		return hmod    
def h_dict( p ):
		"""
		:return: Dictionnary corresponding to the observation operator in the DAPPER format
		"""
		h = { 'M': p,
			'model': def_hmod(),
			'step' : Id_mat(p),
			'noise': modelling.GaussRV(C=std_o * np.eye(p))}
		return h