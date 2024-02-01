import numpy as np
from astropy.stats import FitnessFunc, bayesian_blocks

class ExponentialBlocks_Events(FitnessFunc):
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
