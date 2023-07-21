import numpy as np 
# from scipy.interpolate import interp1d 
from joblib import Parallel, delayed
from jax import numpy as jnp

import logging
log = logging.getLogger(__name__)

try:
    import classy 
    class_present = True
except:
    class_present = False

try:
    import camb 
    camb_present = True
except:
    camb_present = False 


class_paramset = (['T_cmb','h', 'Omega_m', 'Omega_b', 'Omega_k', 'A_s', 'n_s', 'alpha_s', 'r', 'k_pivot','YHe','N_ur', 'N_ncdm','m_ncdm', 'modes','output', 'l_max_scalars'], #, 'l_max_tensors', 'n_t', 'alpha_t',, 'w0',     'wa'],\
                  ['T_cmb','h', 'Omega_m', 'Omega_b', 'Omega_k', 'A_s', 'n_s', 'alpha_s', 'r', 'k_pivot','YHe','N_ur', 'N_ncdm','m_ncdm', 'modes','output', 'l_max_scalars']) #, 'l_max_tensors', 'n_t', 'alpha_t',, 'w0_fld', 'wa_fld'])

camb_paramset = (['T_cmb', 'h', 'Omega_k', 'Omega_b', 'Omega_c', 'A_s', 'n_s', 'alpha_s', 'YHe', 'N_ur',          'N_ncdm',         'm_ncdm','w0','wa'],
                 ['TCMB', 'H0', 'omk',     'ombh2',  'omch2',    'As',  'ns',  'nrun',    'YHe', 'num_nu_massless','num_nu_massive','mnu',   'w', 'wa'])

ccl_paramset = (['Omega_c','Omega_b', 'h', 'A_s', 'sigma8', 'n_s', 'Omega_k', 'Omega_g', 'N_eff', 'm_nu', 'mnu_type', 'w0', 'wa'],
                ['Omega_c','Omega_b', 'h', 'A_s', 'sigma8', 'n_s', 'Omega_k', 'Omega_g', 'N_eff', 'm_nu', 'mnu_type', 'w0', 'wa'])

class cosmology:
    '''
        Sets up Cosmology wrapper for various options of libraries.
    '''

    def __init__(self, backend, **kwargs):
        #-------------------------------------------------------------------------
        # COSMOLOGICAL PARAMETERS (Planck 2018 best fit TT,TE,EE+lowE+lensing)
        # Table 1, First column, Plick best fit [https://arxiv.org/abs/1807.06209]
        #-------------------------------------------------------------------------
        self.params = dict({
## MAIN PARAMETERS ==========================
            'T_cmb': 2.72548,
            'h': 0.6732,
#=================Omegas========================
            'Omega_m': 0.3158,    # Matter density fraction
            'Omega_b': 0.0493890, # baryon density fraction
            'Omega_k': 0.,
#=============Spectral params===================
            'A_s': 2.10058e-9,   # scalar amplitude
            'n_s': 0.96605,      # scalar spectral index
            'alpha_s': 0.,       # scalar spectral running
            'k_pivot': 0.05,     # pivot scale for scalar perturbations
            'r': 0.,             # tensor to scalar ratio
            'n_t': 0.,           # tensor spectral index
            'alpha_t': 0.,       # tensor spectral running
            
#=============reionization========================
            'tau_reion': 0.0543,    # optical depth at reionization
            'YHe': 0.2454,          # Helium fraction
#=============Light relics===========================
            'N_ur': 2.0328,     # Ultra-relativistic species (ultra-light neutrinos)
            'N_ncdm': 1.,       # Light massive relics including massive neutrinos
            'm_ncdm': 0.06,     # mass of non-cold light relics
#===========DE EOS===================================
            'w0': -1.,         
            'wa': 0.,

## DERIVED PARAMETERS =================
            # 'Omega_de': None,   # dark energy fraction
            # 'Omega_c': None,   # CDM fraction
            # 'Omega_rad': None, # Radiation density fraction
            # 'sigma8': None,    # sigma8, not preferred if A_s is provided
            # 'N_eff': None,     # ultrarelativistic and massive neutrino density

#============technical===============================
            'modes': 's t',
            'output': 'tCl pCl lCl mPk',    
            'cosmo_backend': 'CLASS',        
            'l_max_scalars': 4000,
            'l_max_tensors': 1500,
            'lensing': 'yes'})

        if 'load_params' in kwargs:
            pass

        self.params.update(kwargs)

        # if ('A_s' in kwargs) or not('sigma8' in kwargs):
        #     del self.params['sigma8']
        # elif 'sigma8' in kwargs:
        #     del self.params['A_s']

        #==========Derived====================
        if (self.params['cosmo_backend'].upper() == 'CAMB') and not camb_present:
            backend.print2log(log, "CAMB dependency not met. Install CAMB to use CAMB backend.", level="critical")
            exit()
        if (self.params['cosmo_backend'].upper() == 'CAMB') and camb_present:
            self.params['Omega_c'] = self.params['Omega_m'] - self.params['Omega_b']

            self.camb_params = {}
            for i, common_key in enumerate(camb_paramset[0]):
                self.camb_params[camb_paramset[1][i]] = self.params[common_key]
            self.camb_params['ombh2'] *= self.params['h']**2.
            self.camb_params['omch2'] *= self.params['h']**2.
            self.camb_params['HO'] *= 100.

            camb_par = camb.set_params(**self.camb_params)
            self.camb_wsp = camb.get_results(camb_par)



        if (self.params['cosmo_backend'].upper() == 'CLASS') and not class_present:
            backend.print2log(log, "CLASS dependency not met. Install CLASS and Classy to use CLASS backend.", level="critical")
            exit()
        if (self.params['cosmo_backend'].upper() == 'CLASS') and class_present:
            self.class_params = {}
            for i, common_key in enumerate(class_paramset[0]):
                self.class_params[class_paramset[1][i]] = self.params[common_key]

            self.class_wsp = classy.Class()
            self.class_wsp.set(self.class_params)     
            self.class_wsp.compute() 

            self._z_grid = np.logspace(-5, 6, num=10000) 
            self._comoving_dist = jnp.asarray(Parallel(n_jobs=-2, prefer="threads")(delayed (self.class_wsp.comoving_distance)(z) for z in self._z_grid))
            self._growth_factor = jnp.asarray(Parallel(n_jobs=-2, prefer="threads")(delayed (self.class_wsp.scale_independent_growth_factor)(z) for z in self._z_grid))
            self._Hubble = jnp.asarray(Parallel(n_jobs=-2, prefer="threads")(delayed (self.class_wsp.Hubble)(z) for z in self._z_grid))

            self._z_grid = jnp.asarray(self._z_grid)

        # if self.params['cosmo_backend'].upper() == 'CLASS':
        #     _z_grid = np.logspace(-3, 3, num=1000) 
        #     # _comoving_dist= np.empty(self.z_for_comov.shape)
        #     # for i, z in enumerate(_z_for_comov):
        #         # _comoving_dist[i] = self.class_wsp.comoving_distance(z)
        #     _comoving_dist = Parallel(n_jobs=-2, prefer="threads")(delayed (self.class_wsp.comoving_distance)(z) for z in _z_grid)
        #     _growth_factor = Parallel(n_jobs=-2, prefer="threads")(delayed (self.class_wsp.scale_independent_growth_factor)(z) for z in _z_grid)

        # self.__z2comov_interpol = interp1d(_z_grid,_comoving_dist, kind='linear', bounds_error=False, fill_value="extrapolate")
        # self.__comov2z_interpol = interp1d(_comoving_dist,_z_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
        # self.__growthD_interpol = interp1d(_z_grid,_growth_factor, kind='linear', bounds_error=False, fill_value="extrapolate")
        # self.__HubbleH_interpol = interp1d(_z_grid,_Hubble, kind='linear', bounds_error=False, fill_value="extrapolate")

    def comoving_distance(self, z):
        if self.params['cosmo_backend'].upper() == 'CLASS':
            # return self.__z2comov_interpol(z)
            return jnp.interp(z, self._z_grid, self._comoving_dist)
        if self.params['cosmo_backend'].upper() == 'CAMB':
            return self.camb_wsp.comoving_radial_distance(z)

    def comoving_distance2z(self,comoving_distance):
        # return self.__comov2z_interpol(comoving_distance)
        return jnp.interp(comoving_distance, self._comoving_dist, self._z_grid)

    def growth_factor_D(self, z):
        # return self.__growthD_interpol(z)
        return jnp.interp(z, self._z_grid, self._growth_factor)

    def Hubble_H(self, z):
        # return self.__HubbleH_interpol(z)
        return jnp.interp(z, self._z_grid, self._Hubble)



    
