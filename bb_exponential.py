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
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` and ``ncp_fun`` are ignored.
    ncp_fun : function, optional
        If specified, the value of ncp_prior is computed using this function
        with N (=sum(x)) as input
        If ncp_fun is specified ``gamma`` and ``p0`` are ignored
    """
    def __init__(self, p0=0.05, gamma=None, ncp_prior=None, ncp_fun=None):
        super().__init__(p0, gamma, ncp_prior)
        self.ncp_fun = ncp_fun

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
        if self.ncp_fun is not None:
            return self.ncp_fun(N)
        elif self.gamma is not None:
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
        #u = np.arange(140000,2670000,25450) #media degli N
        #y = np.array([
        #    300,700,1050,1300,1550,1800,2050,2300,2400,2600,
        #    2950,3000,3200,3300,3700,3800,4000,4150,4200,4400,
        #    4550,4700,4800,5000,5100,5300,5250,5400,5600,5700,
        #    5900,5900,6100,6300,6300,6350,6350,6500,6500,6700,
        #    6600,6650,6650,6650,6700,6750,6750,6750,6750,6850,
        #    6850,6850,6900,6900,6900,7000,7000,7000,7000,7000,
        #    7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
        #    7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
        #    7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,
        #    7000,7000,7000,7000,7000,7000,7000,7000,7000,7000  ])#ncp_prior per ottimizzarlo
        #return scipy.interpolate.CubicSpline(u, y)(N)
        u_y = np.array([(100000, 200), (110000, 225), (120000, 270), (130000, 290), (140000, 320), (150000, 400), (160000, 615.2451387849095), (170000, 769.0426073626077), (180000, 912.4425305120708), (190000, 1039.5592995695006), (200000, 1146.6567688625566), (210000, 1241.275883981712), (220000, 1334.5308572856504), (230000, 1532.1128285531624), (240000, 1631.983249843333), (250000, 1731.6887098203943), (260000, 1879.9999384279372), (270000, 2026.227688353349), (280000, 2171.731710805428), (290000, 2321.5849606132467), (300000, 2429.526794729129), (310000, 2533.2305757248723), (320000, 2711.6803186474867), (330000, 2756.2524255234966), (340000, 2787.366888764701), (350000, 2829.1048274962022), (360000, 2900), (370000, 2912.9822156774767), (380000, 3066.4999740577014), (390000, 3207.342876184784), (400000, 3276.958008987681), (410000, 3284.8418908912945), (420000, 3300), (430000, 3372.2946071985752), (440000, 3463.523140623286), (450000, 3517.111630523897), (460000, 3536.3717122166304), (470000, 3591.3272473974002), (480000, 3734.6707198886857), (490000, 3911.577851531317), (500000, 4033.9108453381855), (510000, 4074.8817914281317), (520000, 4093.9870014880653), (530000, 4048.327925644581), (540000, 4234.4495001727046), (550000, 4323.6880261325523), (560000, 4395.198916129244), (570000, 4442.2803859913665), (580000, 4462.134384895589), (590000, 4474.256346571001), (600000, 4510.031402086732), (610000, 4583.934618571861), (620000, 4671.375600599834), (630000, 4744.357812704114), (640000, 4800.711755730153), (650000, 4855.9151801188655), (660000, 4818.958879392718), (670000, 4978.54217763008), (680000, 5020.787753185267), (690000, 5052.960534254824), (700000, 5100), (710000, 5179.487815685128), (720000, 5264.326232528083), (730000, 5321.3479882104775), (740000, 5351.687985762398), (750000, 5394.764551060448), (760000, 5479.103851336002), (770000, 5567.8650523262495), (780000, 5604.511360541746), (790000, 5579.951974729595), (800000, 5550.942520175716), (810000, 5570.8320952179765), (820000, 5637.4999838380745), (830000, 5726.016212786224), (840000, 5814.414763873945), (850000, 5986.086980921798), (860000, 5928.398211497241), (870000, 5959.5606806436945), (880000, 6014.006373544878), (890000, 6102.682117475), (900000, 6183.059489732281), (910000, 6210.087472361603), (920000, 6198.399085055407), (930000, 6202.86606922961), (940000, 6260.57385312554), (950000, 6354.317169401872), (960000, 6457.210656236934), (970000, 6545.9856583129895), (980000, 6600), (990000, 6610.103076399633), (1000000, 6600.789201352866), (1010000, 6606.05298723126), (1020000, 6630.92359116326), (1030000, 6649.6098170607165), (1040000, 6643.442313423734), (1050000, 6637.156166762502), (1060000, 6668.222478491959), (1070000, 6737.412551555299), (1080000, 6795.236172274472), (1090000, 6800.030735452649), (1100000, 6785.918209921648), (1110000, 6817.2031475941785), (1120000, 6908.96915612123), (1130000, 6990.427537505028), (1140000, 6992.516558884732), (1150000, 6937.684034954056), (1160000, 6895.999721237238), (1170000, 6906.866325140376), (1180000, 6940.339239319563), (1190000, 6959.582814603485), (1200000, 6957.993934610502), (1210000, 6949.088081971744), (1220000, 6944.046575489862), (1230000, 6946.26379922899), (1240000, 6957.567265694911), (1250000, 6976.842434537024), (1260000, 7000), (1270000, 7024.518897614738), (1280000, 7043.7113889813145), (1290000, 7052.803776289019), (1300000, 7052.536867324852), (1310000, 7050.062932294076), (1320000, 7051.01993639146), (1330000, 7052.02517872621), (1340000, 7047.138738380322), (1350000, 7039.8703227876495), (1360000, 7046.500953373964), (1370000, 7079.932640040183), (1380000, 7125.175870289963), (1390000, 7156.252166047541), (1400000, 7161.268609698072), (1410000, 7152.5836808043605), (1420000, 7144.1614455502795), (1430000, 7143.134700131197), (1440000, 7153.098070467249), (1450000, 7173.477335306202), (1460000, 7174.590302875795), (1470000, 7186.236341919776), (1480000, 7186.484123178069), (1490000, 7188.835930978074), (1500000, 7189.447332862699), (1510000, 7191.750217934312), (1520000, 7189.202719966802), (1530000, 7163.995593289019), (1540000, 7200.606551112646), (1550000, 7210.654181870235), (1560000, 7204.263540529191), (1570000, 7197.780027132758), (1580000, 7197.647902911536), (1590000, 7199.879932764492), (1600000, 7200.766219223369), (1610000, 7200.364404404361), (1620000, 7199.860794943026), (1630000, 7199.820239461689), (1640000, 7199.978400427554), (1650000, 7200.054354956158), (1660000, 7200.030298082164), (1670000, 7199.991757326904), (1680000, 7199.9864276463195), (1690000, 7199.997445678717), (1700000, 7200.003797022601), (1710000, 7200.002464048843), (1720000, 7199.999556927054), (1730000, 7199.998987230106), (1740000, 7199.9997406203265),(1750000, 7200.000260529659), (1760000, 7200.000196733262), (1770000, 7199.99998060138), (1780000, 7199.999925304678), (1790000, 7199.999975725942), (1800000, 7200.00001749045), (1810000, 7200.000015459508), (1820000, 7199.999999629206), (1830000, 7199.999994556318), (1840000, 7199.999997843628), (1850000, 7200.00000114195), (1860000, 7200.0000011977445), (1870000, 7200.000000056484), (1880000, 7199.999999608222), (1890000, 7199.9999998153735), (1900000, 7200.000000071788), (1910000, 7200.000000091602), (1920000, 7200.000000010658), (1930000, 7199.999999972186), (1940000, 7199.999999984629), (1950000, 7200.000000004267), (1960000, 7200.00000000692), (1970000, 7200.000000001277), (1980000, 7199.999999998055), (1990000, 7199.9999999987485), (2000000, 7200.000000000232), (2010000, 7200.000000000517), (2020000, 7200.000000000131), (2030000, 7199.999999999866), (2040000, 7199.9999999999), (2050000, 7200.00000000001), (2060000, 7200.000000000038), (2070000, 7200.000000000012), (2080000, 7199.999999999991), (2090000, 7199.999999999992), (2100000, 7200.0), (2110000, 7200.000000000003), (2120000, 7200.000000000001), (2130000, 7199.999999999999), (2140000, 7199.999999999999), (2150000, 7200.0), (2160000, 7200.0), (2170000, 7200.0), (2180000, 7200.0), (2190000, 7200.0), (2200000, 7200.000000000001), (2210000, 7200.0), (2220000, 7200.0), (2230000, 7200.0), (2240000, 7200.0), (2250000, 7200.0), (2260000, 7200.0), (2270000, 7200.0), (2280000, 7200.0), (2290000, 7200.0), (2300000, 7200.0), (2310000, 7200.0), (2320000, 7200.0), (2330000, 7200.0), (2340000, 7200.0), (2350000, 7200.0), (2360000, 7200.0), (2370000, 7200.0), (2380000, 7200.0), (2390000, 7200.0), (2400000, 7200.0), (2410000, 7200.0), (2420000, 7200.0), (2430000, 7200.0), (2440000, 7200.0), (2450000, 7200.0), (2460000, 7200.0), (2470000, 7200.0), (2480000, 7200.0), (2490000, 7200.0), (2500000, 7200.0), (2510000, 7200.0), (2520000, 7200.0), (2530000, 7200.0), (2540000, 7200.0), (2550000, 7200.0), (2560000, 7200.0), (2570000, 7200.0), (2580000, 7200.0), (2590000, 7200.0), (2600000, 7200.0), (2610000, 7200.0), (2620000, 7200.0), (2630000, 7200.0), (2640000, 7200.0), (2650000, 7200.0), (2660000, 7200.0), (2670000, 7200.0), (2680000, 7200.0), (2690000, 7200.0), (2700000, 7200.0), (2710000, 7200.0), (2720000, 7200.0), (2730000, 7200.0), (2740000, 7200.0), (2750000, 7200.0), (2760000, 7200.0), (2770000, 7200.0), (2780000, 7200.0), (2790000, 7200.0), (2800000, 7200.0), (2810000, 7200.0), (2820000, 7200.0), (2830000, 7200.0), (2840000, 7200.0), (2850000, 7200.0), (2860000, 7200.0), (2870000, 7200.0), (2880000, 7200.0), (2890000, 7200.0), (2900000, 7200.0), (2910000, 7200.0), (2920000, 7200.0), (2930000, 7200.0), (2940000, 7200.0), (2950000, 7200.0), (2960000, 7200.0), (2970000, 7200.0), (2980000, 7200.0), (2990000, 7200.0), (3000000, 7200.0)])
        return scipy.interpolate.CubicSpline(u_y[:,0], u_y[:,1])(N)
    
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
