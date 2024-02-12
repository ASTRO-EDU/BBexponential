import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import Events, bayesian_blocks
import scipy

class ExponentialBlocks_Events_Deprecated(Events):
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

class ExponentialBlocks_Events_Deprecated_Alt(Events):
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
    
    def fit(self, t, x=None, sigma=None):
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array-like
            data times (one dimensional, length N)
        x : array-like, optional
            data values
        sigma : array-like or float, optional
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """
        t, x, sigma = self.validate_input(t, x, sigma)

        # compute values needed for computation, below
        if "a_k" in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma**2
        if "b_k" in self._fitness_args:
            bk_raw = x / sigma**2
        if "c_k" in self._fitness_args:
            ck_raw = x * x / sigma**2

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # Compute ncp_prior if not defined
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(sum(x))
        else:
            ncp_prior = self.ncp_prior

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):
            # Compute fit_vec : fitness of putative last block (end at R)
            kwds = {}

            # T_k: width/duration of each block
            if "T_k" in self._fitness_args:
                kwds["T_k"] = block_length[: (R + 1)] - block_length[R + 1]

            # N_k: number of elements in each block
            if "N_k" in self._fitness_args:
                kwds["N_k"] = np.cumsum(x[: (R + 1)][::-1])[::-1]

            # a_k: eq. 31
            if "a_k" in self._fitness_args:
                kwds["a_k"] = 0.5 * np.cumsum(ak_raw[: (R + 1)][::-1])[::-1]

            # b_k: eq. 32
            if "b_k" in self._fitness_args:
                kwds["b_k"] = -np.cumsum(bk_raw[: (R + 1)][::-1])[::-1]

            # c_k: eq. 33
            if "c_k" in self._fitness_args:
                kwds["c_k"] = 0.5 * np.cumsum(ck_raw[: (R + 1)][::-1])[::-1]

            # evaluate fitness function
            fit_vec = self.fitness(**kwds)

            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]

            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]

        # ----------------------------------------------------------------
        # Now find changepoints by iteratively peeling off the last block
        # ----------------------------------------------------------------
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while i_cp > 0:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        if i_cp == 0:
            change_points[i_cp] = 0
        change_points = change_points[i_cp:]

        return edges[change_points]
    
    
    def compute_ncp_prior(self, N):
        """
        If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
        or ``p0``.
        """
        if self.gamma is not None:
            return -np.log(self.gamma)
        elif self.p0 is not None:
            return self.p0_prior(N)
        else:
            raise ValueError(
                "``ncp_prior`` cannot be computed as neither "
                "``gamma`` nor ``p0`` is defined."
            )
            
    def p0_prior(self, N):
        """Empirical prior, parametrized by the false alarm probability ``p0``.
        """
        def model_pow2(x,u):
            return np.divide(x[0]*u**3+x[1]*u**2+x[2]*u+x[3],u**3+x[4]*u**2+x[5]*u+x[6])
        opt_par = [6.10571861e+03, -2.06016282e+08, -1.00880302e+09, -5.57425062e+03, -1.87499473e+04,  9.99999996e+09, 2.37041761e+07]
        
        u = np.arange(140000,2670000,25450) #media degli N
        y = np.array([
            300,700,1050,1300,1550,1800,2050,2300,2400,2600,
            2950,3000,3200,3300,3700,3800,4000,4150,4200,4400,
            4550,4700,4800,5000,5100,5300,5250,5400,5600,5700,
            5900,5900,6100,6300,6300,6350,6350,6500,6500,6700,
            6600,6650,6650,6650,6700,6750,6750,6750,6750,6850,
            6850,6850,6900,6900,6900,7000,7000,7000,7000,7000,
            7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
            7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
            7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
            7000,7000,7000,7000,7000,7000,7000,7000,7000,7000  ])#ncp_prior per ottimizzarlo
        return scipy.interpolate.CubicSpline(u, y)(N)

        #return model_pow2(opt_par,N)
        #return 7000-np.exp(-1e-6*4.1*(N-2260517)) #old version
    
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
        edges = np.concatenate([np.array([edge_l]), 0.5 * (t_new[1:] + t_new[:-1]), np.array([edge_r])])
        block_length = t_new[-1] - edges
        T_k = block_length[:-1] - block_length[-1]
        N_k = np.cumsum(x_new[::-1])[::-1]
        S_k = -(1/N_k)*(np.cumsum((T_k*self.anti_N_k(N_k))[::-1])[::-1])
        return T_k,N_k,S_k
    
    
def test_bb_exp(t,x,fitness = ExponentialBlocks_Events,save = False,name = 'out.png', **kwargs):
    plt.step(t, x)
    t_start = time.time()
    xcoords = bayesian_blocks(t,x,fitness=fitness, **kwargs)
    t_used = time.time()-t_start
    print(f'Tempo di Calcolo: {t_used}')
    for xc in xcoords:
        plt.axvline(x=xc, color='grey', alpha=0.3)
    print(f'Estremi: {xcoords}')
    for i in range(len(xcoords)-1):
        edge_l = xcoords[i]
        edge_r = xcoords[i+1]
        params = fitness(**kwargs).get_parameters(edge_l,edge_r,t,x)
        a = params['a']
        gamma = params['gamma']
        print(f'Blocco {i}: a={a},gamma={gamma}')
        plot_t = np.linspace(edge_l,edge_r,1000)
        plot_x = gamma * np.exp(a*(plot_t - edge_r))
        plt.plot(plot_t,plot_x)
    if save:
        plt.savefig(name)
    plt.show()