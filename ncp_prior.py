import numpy as np
from astropy.stats import bayesian_blocks
from bb_exponential import ExponentialBlocks_Events_Alt_2
from tqdm.auto import tqdm
import os
from itertools import repeat
from multiprocessing import Pool
import csv

def generator(t,gamma):
    while True:
        x = np.random.poisson(125,len(t))
        idx = np.random.poisson(100)
        y = np.concatenate([np.zeros(idx),np.floor(gamma*np.exp(-np.linspace(0,20,1000)))[:1000-idx]])
        x=x+y
        yield x

def num_blocks(t,x,ncp_prior_list,fitness):
    arr = np.zeros_like(ncp_prior_list)
    for i in range(len(ncp_prior_list)):
        edg = bayesian_blocks(t,x,fitness=fitness,ncp_prior=ncp_prior_list[i])
        n_edges = len(edg)
        n_blocks = n_edges - 1
        #print(n_blocks)
        arr[i] = n_blocks
    return arr

def ncp_optimizer(t,gamma,ncp_prior_list,fitness,n_iter=10):
    gen = generator(t,gamma)
    arr = np.zeros((n_iter,len(ncp_prior_list)))
    for i in tqdm(range(n_iter)):
        x = next(gen)
        arr[i] = num_blocks(t,x,ncp_prior_list,fitness)
    arr[arr != 3] = 1
    arr[arr == 3] = 0
    mean_array = np.mean(arr,axis=0)
    return mean_array

def mult_helper(t,gamma,ncp_prior_list,fitness,n_iter):
    #print(gamma)
    p_0_array = ncp_optimizer(t,gamma,ncp_prior_list,fitness = fitness,n_iter=n_iter)
    return np.concatenate([np.array([gamma]),p_0_array,np.array([n_iter]),ncp_prior_list])

def calc_p_0(data,outfile = 'ncp.csv'):
    with Pool(20) as p:
        results = p.starmap(mult_helper,data)
    print(results)
    with open('ncp.csv','a+') as file:
        write = csv.writer(file)
        write.writerows(results)

def construct_data(ncp_prior_list,t,n_iter,gamma):
    return zip(repeat(t),
               repeat(gamma),
               np.array_split(ncp_prior_list,20),
               repeat(ExponentialBlocks_Events_Alt_2),
               repeat(n_iter))

if __name__ == '__main__':
    ncp_gamma_pairs = [(300,np.linspace(300,400,101)),
                       (400,np.linspace(400,500,101)),
                       (500,np.linspace(500,600,101)),
                       (600,np.linspace(550,650,101)),
                       (700,np.linspace(650,750,101)),
                       (800,np.linspace(700,800,101)),
                       (900,np.linspace(800,900,101)),
                       (1000,np.linspace(900,1000,101)),
                       (1100,np.linspace(950,1050,101)),
                       (1200,np.linspace(1050,1150,101)),
                       (1300,np.linspace(1100,1200,101)),
                       (1400,np.linspace(1200,1300,101)),
                       (1500,np.linspace(1300,1400,101)),
                       (1600,np.linspace(1400,1500,101)),
                       (1700,np.linspace(1450,1550,101)),
                       (1800,np.linspace(1550,1650,101)),
                       (1900,np.linspace(1600,1700,101)),
                       (2000,np.linspace(1700,1800,101)),]

    for pair in ncp_gamma_pairs:
	    data = construct_data(ncp_prior_list = pair[1], 
	                          t = np.arange(1000), 
	                          n_iter = 500,
	                          gamma = pair[0])
	    calc_p_0(data,'ncp.csv')
