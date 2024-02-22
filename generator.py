import numpy as np
import scipy
from itertools import repeat

class gammaflash_generator:
    '''
    generates gammaflash-like signal
    
    Parameters
    t_s        : array-like, list of starting times of gammaflashes
    H_s        : float or array-like, list of amplitudes H of gammaflashes
    sigma_s    : float or array-like, list of smoothing factor of gammaflashes
    tau1_s     : float or array-like, list of decay time constant of fast components of gammaflashes
    tau2_s     : float or array-like, list of decay time constant of slow components of gammaflashes
    background : float, mean background level
    
    Methods
    signal_get : returns the realization of the gammaflashes with time and data arrays t_plot and x_plot
    signal_get_noise : same as above but adds a gaussian noise that is rms*max(signal)
        both with param
             t_plot     : array, list of times to be plotted
    '''
    def __init__(self,t_s,H_s,sigma_s,tau1_s,tau2_s,background):
        if not hasattr(H_s,'__iter__'):
            H_s = repeat(H_s)
        if not hasattr(sigma_s,'__iter__'):
            sigma_s = repeat(sigma_s)
        if not hasattr(tau1_s,'__iter__'):
            tau1_s = repeat(tau1_s)
        if not hasattr(tau2_s,'__iter__'):
            tau2_s = repeat(tau2_s)
        self.t_s = t_s
        self.H_s = H_s
        self.sigma_s = sigma_s
        self.tau1_s = tau1_s
        self.tau2_s = tau2_s
        self.background = background
    def signal_get(self,t_plot):
        x_plot = self.background*np.ones_like(t_plot)
        for t_start,H,sigma,tau_1,tau_2 in zip(self.t_s,self.H_s,self.sigma_s,self.tau1_s,self.tau2_s):
            x_plot += self._generator(H,sigma,tau_1,tau_2)(t_plot-t_start)
        return x_plot
    def signal_get_noise(self,t_plot,sigma):
        x_plot = self.signal_get(t_plot)
        return np.random.normal(x_plot,sigma)
    def _generator(self,H,sigma,tau_1,tau_2):
        return (lambda t: H*(np.exp((sigma**2-2*t*tau_1)/(2*tau_1**2))*
                        (1+scipy.special.erf(t/(np.sqrt(2)*sigma)-(np.sqrt(2)*sigma)/(2*tau_1)))
                        -np.exp((sigma**2-2*t*tau_2)/(2*tau_2**2))*
                        (1+scipy.special.erf(t/(np.sqrt(2)*sigma)-(np.sqrt(2)*sigma)/(2*tau_2)))))