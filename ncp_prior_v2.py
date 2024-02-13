import numpy as np
from astropy.stats import bayesian_blocks
from bb_exponential import ExponentialBlocks_Events
from tqdm.auto import tqdm
import os
from itertools import repeat
from multiprocessing import Pool
import csv
from reader_class import Reader

def num_blocks(t,x,ncp_prior_list,fitness):
    arr = np.zeros_like(ncp_prior_list)
    for i in range(len(ncp_prior_list)):
        edg = bayesian_blocks(t,x,fitness=fitness,ncp_prior=ncp_prior_list[i])
        n_edges = len(edg)
        n_blocks = n_edges - 1
        arr[i] = n_blocks
    return arr

def ncp_optimizer(t,data_bracket,ncp_prior_list,fitness):
    arr = np.zeros((len(data_bracket),len(ncp_prior_list)))
    for i in tqdm(range(len(data_bracket))):
        x = data_bracket[i]
        arr[i] = num_blocks(t,x,ncp_prior_list,fitness)
    arr[arr != 3] = 1
    arr[arr == 3] = 0
    mean_array = np.mean(arr,axis=0)
    return mean_array

def mult_helper(t,data_bracket,ncp_prior_list,fitness):
    p_0_array = ncp_optimizer(t,data_bracket,ncp_prior_list,fitness = fitness)
    return [p_0_array,ncp_prior_list]

def calc_p_0(data,outfile = 'ncp.csv'):
    with Pool(50) as p:
        results = p.starmap(mult_helper,data)
    print(results)
    firstrow = np.concatenate([result[1] for result in results])
    secondrow = np.concatenate([result[0] for result in results])
    with open('ncp.csv','a+') as file:
        write = csv.writer(file)
        write.writerow(firstrow)
        write.writerow(secondrow)
        
        
def data_raw_to_data_bracket_list(data,nbins=100):
    N_list = [sum(data[i][1000:2000]) for i in range(len(data))]
    passo = (max(N_list)-min(N_list))*(1/nbins)
    estr_inf = min(N_list)
    a = [[data[i][1000:2000] for i in range(len(N_list)) if estr_inf+n*passo<=N_list[i]<=estr_inf+(n+1)*passo] for n in range(nbins)]
    return a,estr_inf,passo
    
def construct_data(ncp_prior_list,t,data_bracket):
    return zip(repeat(t),
               repeat(data_bracket),
               np.array_split(ncp_prior_list,20),
               repeat(ExponentialBlocks_Events))

if __name__ == '__main__':
    
    filename = '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg0/dl0/wf_runId_00293_configId_00000_2023-01-27T14_32_51.090204.h5'
    
    data = Reader().get_data_list(filename,0,10000)
    
    a,estr_inf,passo = data_raw_to_data_bracket_list(data,nbins=100)

    ncp_list = [np.linspace(350,450,101),
                np.linspace(600,700,101),
                np.linspace(950,1050,101),
                np.linspace(1200,1300,101),
                np.linspace(1450,1550,101),
                np.linspace(1700,1800,101),
                np.linspace(1950,2050,101),
                np.linspace(2200,2300,101),
                np.linspace(2300,2400,101),
                np.linspace(2500,2600,101),
                np.linspace(2850,2950,101),
                np.linspace(2900,3000,101),
                np.linspace(3100,3200,101),
                np.linspace(3200,3300,101),
                np.linspace(3600,3700,101),
                np.linspace(3700,3800,101),
                np.linspace(3900,4000,101),
                np.linspace(4050,4150,101),
                np.linspace(4100,4200,101),
                np.linspace(4300,4400,101),
                np.linspace(4450,4550,101),
                np.linspace(4600,4700,101),
                np.linspace(4700,4800,101),
                np.linspace(4900,5000,101),
                np.linspace(5000,5100,101),
                np.linspace(5200,5300,101),
                np.linspace(5150,5250,101),
                np.linspace(5300,5400,101),
                np.linspace(5500,5600,101),
                np.linspace(5600,5700,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(6000,6100,101),
                np.linspace(6200,6300,101),
                np.linspace(6200,6300,101),
                np.linspace(6250,6350,101),
                np.linspace(6250,6350,101),
                np.linspace(6400,6500,101),
                np.linspace(6400,6500,101),
                np.linspace(6600,6700,101),
                np.linspace(6500,6600,101),
                np.linspace(6550,6650,101),
                np.linspace(6550,6650,101),
                np.linspace(6550,6650,101),
                np.linspace(6600,6700,101),
                np.linspace(6650,6750,101),
                np.linspace(6650,6750,101),
                np.linspace(6650,6750,101),
                np.linspace(5800,5900,101),
                np.linspace(6750,6850,101),
                np.linspace(5800,5900,101),
                np.linspace(6750,6850,101),
                np.linspace(6800,6900,101),
                np.linspace(6800,6900,101),
                np.linspace(6800,6900,101),
                np.linspace(6900,7000,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(7500,7600,101),
                np.linspace(6850,6950,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(7450,7550,101),
                np.linspace(8350,8450,101),
                np.linspace(7800,7900,101),
                np.linspace(5800,5900,101),
                np.linspace(8050,8150,101),
                np.linspace(9500,9600,101),
                np.linspace(9000,9100,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(6900,7000,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(6900,7000,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(6900,7000,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(5800,5900,101),
                np.linspace(6800,6900,101)]
    
    ncp_x_pairs = [(a[i][:1000],ncp_list[i]) for i in range(len(a))]
    
    with open('ncp.csv','a+') as file:
        write = csv.writer(file)
        write.writerow([estr_inf,passo])
    
    for pair in ncp_x_pairs:
        data = construct_data(ncp_prior_list = pair[1], 
                              t = np.arange(1000), 
                              data_bracket = pair[0])
        calc_p_0(data,'ncp.csv')
