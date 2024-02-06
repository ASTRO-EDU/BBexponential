import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import Events, bayesian_blocks
import scipy

class ExponentialBlocks_Events(Events):
    r""" Bayesian Blocks Fitness for binned or unbinned events with piecewise exponential function over the blocks

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def anti_N_k(self,N_k):
        x = np.zeros(len(N_k))
        x[-1]=N_k[-1]
        for i in range(len(N_k)-1):
            x[i] = N_k[i]-N_k[i+1]
        return x

    def compute_a(self,T_k,N_k,S_k):
        """Computes as instructed by eq. C116 in Scargle (2013).

        Parameters
        ---------
        N_k : array-like
            number of events in blocks in receding order (from last to first)
        T_k : array-like, float
            length of the blocks in decreasing order
        ---------

        Returns
        ---------
        a : array-like
            optimal parameter for each block identified by T_k
        """
        a = np.ones_like(T_k,dtype=float)
        f = 1
        i = 1
        #print(T_k,N_k,S_k)
        #implementation of Newton's method to find optimal a
        while i<100 and np.any(np.abs(f)>1e-8):
            #Q_k defined as in C114 of Scargle (2013)
            Q_k = np.exp(-a*T_k)/(1-np.exp(-a*T_k))
            #f defined as in C109 of Scargle (2013)
            f = (1/a) - T_k*Q_k + S_k
            #f_prime defined as in C113 of Scargle (2013)
            f_prime = -(1/(a*a)) + T_k*T_k*Q_k*(1+Q_k)
            a -= np.divide(f,f_prime)
            #print(i,max(np.abs(f)),a)
            #print(f'a:{a}')
            i += 1
        #print(i,max(np.abs(f)))
        return a

    def fitness(self,T_k,N_k):
        #S_k defined as in C110 of Scargle (2013)
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        a = self.compute_a(T_k,N_k,S_k)
        # the log (to have additivity of the blocks) of C105 from Scargle (2013)
        likelihood = N_k * np.log((a*N_k)/(1-np.exp(-a*T_k))) + a*N_k*S_k - N_k
        return np.log(likelihood)

    def get_parameters(self,edge_l,edge_r,t,x):
        T_k,N_k,S_k = self.get_T_k_N_k_S_k_from_edges(edge_l,edge_r,t,x)
        a = self.compute_a(T_k,N_k,S_k)
        gamma = np.divide(a*N_k,1-np.exp(-a*T_k))
        return {'a':a[0],'gamma':gamma[0]}

    def get_T_k_N_k_S_k_from_edges(self,edge_l,edge_r,t,x):
        flagStart=0
        flagEnd=0
        t_end_index=0
        for i,t_d in enumerate(t):
            if edge_l<=t_d and t_d<=edge_r:
                if flagStart==0:
                    flagStart=1
                    t_start_index=i
            elif t_d>edge_r and flagEnd==0:
                flagEnd=1
                t_end_index=i
        if t_end_index == 0:
            t_end_index = len(t)
        t_new = t[t_start_index:t_end_index]
        x_new = x[t_start_index:t_end_index]
        #print(t_new)
        #print(x_new)
        edges = np.concatenate([np.array([edge_l]), 0.5 * (t_new[1:] + t_new[:-1]), np.array([edge_r])])
        #print(edges)
        #print(len(edges))
        block_length = t_new[-1] - edges
        #print(block_length)
        T_k = block_length[:-1] - block_length[-1]
        N_k = np.cumsum(x_new[::-1])[::-1]
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        return T_k,N_k,S_k

class ExponentialBlocks_Events_Alt(Events):
    r""" Bayesian Blocks Fitness for binned or unbinned events with piecewise exponential function over the blocks
    
    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def anti_N_k(self,N_k):
        x = np.zeros(len(N_k))
        x[-1]=N_k[-1]
        for i in range(len(N_k)-1):
            x[i] = N_k[i]-N_k[i+1]
        return x
    
    def fitness(self, N_k, T_k):
        # the log (to have additivity of the blocks) of C105 from Scargle (2013)
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        arr=np.array([])
        for n,t,s in zip(N_k,T_k,S_k):
            def neg_likelihood(a):
                return -n*(np.log((a*n)/(1-np.exp(-a*t)))+a*s-1)
            arr=np.append(arr,-scipy.optimize.minimize_scalar(neg_likelihood).fun)
        #likelihood = n*(np.log((a*n)/(1-np.exp(-a*t)))+a*s-1)
        return arr
    
    def get_parameters(self,edge_l,edge_r,t,x):
        T_k,N_k,S_k = self.get_T_k_N_k_S_k_from_edges(edge_l,edge_r,t,x)
        t,n,s = T_k[0],N_k[0],S_k[0]
        def neg_likelihood(a):
            return -n*(np.log((a*n)/(1-np.exp(-a*t)))+a*s-1)
        a = scipy.optimize.minimize_scalar(neg_likelihood).x
        #likelihood = n*(np.log((a*n)/(1-np.exp(-a*t)))+a*s-1)
        gamma = np.divide(a*n,1-np.exp(-a*t))
        return {'a':a,'gamma':gamma}

    def get_T_k_N_k_S_k_from_edges(self,edge_l,edge_r,t,x):
        flagStart=0
        flagEnd=0
        t_end_index=0
        for i,t_d in enumerate(t):
            if edge_l<=t_d and t_d<=edge_r:
                if flagStart==0:
                    flagStart=1
                    t_start_index=i
            elif t_d>edge_r and flagEnd==0:
                flagEnd=1
                t_end_index=i
        if t_end_index == 0:
            t_end_index = len(t)
        t_new = t[t_start_index:t_end_index]
        x_new = x[t_start_index:t_end_index]
        #print(t_new)
        #print(x_new)
        edges = np.concatenate([np.array([edge_l]), 0.5 * (t_new[1:] + t_new[:-1]), np.array([edge_r])])
        #print(edges)
        #print(len(edges))
        block_length = t_new[-1] - edges
        #print(block_length)
        T_k = block_length[:-1] - block_length[-1]
        N_k = np.cumsum(x_new[::-1])[::-1]
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        return T_k,N_k,S_k
    
class ExponentialBlocks_Events_Alt_2(Events):
    r""" Bayesian Blocks Fitness for binned or unbinned events with piecewise exponential function over the blocks
    
    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def anti_N_k(self,N_k):
        x = np.zeros(len(N_k))
        x[-1]=N_k[-1]
        for i in range(len(N_k)-1):
            x[i] = N_k[i]-N_k[i+1]
        return x
    
    def compute_a(self,T_k,N_k,S_k):
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        def Q(a):
            tmp = np.clip(a*T_k,-70,70)
            return np.exp(-tmp)/(1-np.exp(-tmp))
        def f(a):
            return (1/a)-T_k*Q(a)+S_k
        def f_prime(a):
            Q_k=Q(a)
            return -(1/(a**2))+(T_k**2)*Q_k*(1+Q_k)
        def f_second(a):
            Q_k = Q(a)
            return 2/(a**3)-(T_k**3)*Q_k*(1+Q_k)*(1+2*Q_k)
        try:
            a = scipy.optimize.newton(f,np.ones_like(T_k),fprime=f_prime,fprime2=f_second)
        except Exception as e:
            print('Occhio')
            a=1
        return a
        
    def fitness(self, N_k, T_k):
        # C105 from Scargle (2013) (even if it was called likelihood that is an error, it is the log-likelihood)
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        a = self.compute_a(T_k,N_k,S_k)
        return N_k*(np.log((a*N_k)/(1-np.exp(-a*T_k)))+a*S_k-1)
    
    def get_parameters(self,edge_l,edge_r,t,x):
        T_k,N_k,S_k = self.get_T_k_N_k_S_k_from_edges(edge_l,edge_r,t,x)
        a = self.compute_a(T_k,N_k,S_k)
        gamma = np.divide(a*N_k,1-np.exp(-a*T_k))
        try:
            a[0]
            a_array=True
        except Exception as e:
            a_array=False
        if a_array:
            return {'a':a[0],'gamma':gamma[0]}
        return {'a':a,'gamma':gamma}

    def get_T_k_N_k_S_k_from_edges(self,edge_l,edge_r,t,x):
        flagStart=0
        flagEnd=0
        t_end_index=0
        for i,t_d in enumerate(t):
            if edge_l<=t_d and t_d<=edge_r:
                if flagStart==0:
                    flagStart=1
                    t_start_index=i
            elif t_d>edge_r and flagEnd==0:
                flagEnd=1
                t_end_index=i
        if t_end_index == 0:
            t_end_index = len(t)
        t_new = t[t_start_index:t_end_index]
        x_new = x[t_start_index:t_end_index]
        #print(t_new)
        #print(x_new)
        edges = np.concatenate([np.array([edge_l]), 0.5 * (t_new[1:] + t_new[:-1]), np.array([edge_r])])
        #print(edges)
        #print(len(edges))
        block_length = t_new[-1] - edges
        #print(block_length)
        T_k = block_length[:-1] - block_length[-1]
        N_k = np.cumsum(x_new[::-1])[::-1]
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        #print(T_k,N_k,S_k)
        return T_k,N_k,S_k
    
    
def test_bb_exp(t,x,fitn = ExponentialBlocks_Events,save = False,name = 'out.png', **kwargs):
    plt.step(t, x)
    t_start = time.time()
    xcoords = bayesian_blocks(t,x,fitness=fitn, **kwargs)
    t_used = time.time()-t_start
    print(f'Tempo di Calcolo: {t_used}')
    for xc in xcoords:
        plt.axvline(x=xc, color='grey', alpha=0.3)
    print(f'Estremi: {xcoords}')
    for i in range(len(xcoords)-1):
        edge_l = xcoords[i]
        edge_r = xcoords[i+1]
        params = fitn(**kwargs).get_parameters(edge_l,edge_r,t,x)
        a = params['a']
        gamma = params['gamma']
        print(f'Blocco {i}: a={a},gamma={gamma}')
        plot_t = np.linspace(edge_l,edge_r,1000)
        plot_x = gamma * np.exp(a*(plot_t - edge_r))
        plt.plot(plot_t,plot_x)
    if save:
        plt.savefig(name)
    plt.show()