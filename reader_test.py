from reader_class import Reader
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
from bb_exponential import ExponentialBlocks_Events, test_bb_exp
from multiprocessing import Pool
import random
from tqdm.auto import tqdm
import os
import csv
import time
import scipy
from itertools import repeat

filenames = ['/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg0/dl0/wf_runId_00293_configId_00000_2023-01-27T14_32_51.090204.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg0/dl0/wf_runId_00293_configId_00000_2023-02-04T03_08_44.553806.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg0/dl0/wf_runId_00293_configId_00000_2023-02-05T09_38_10.610944.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg1/dl0/wf_runId_00182_configId_00000_2023-02-04T08_47_28.472848.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_01_27/rpg1/dl0/wf_runId_00182_configId_00000_2023-02-09T15_06_11.925745.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg0/dl0/wf_runId_00300_configId_00001_2023-03-24T15_23_42.695318.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg0/dl0/wf_runId_00300_configId_00001_2023-03-28T22_50_18.164402.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg1/dl0/wf_runId_00186_configId_00001_2023-03-24T13_37_23.947960.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg1/dl0/wf_runId_00186_configId_00001_2023-03-30T10_26_22.002155.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-04-30T14_03_22.256890.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-04-30T14_09_49.429046.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-04-30T14_16_21.687428.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-04-30T14_40_13.700898.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-02T06_38_12.862147.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-02T06_44_37.077783.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-03T03_34_43.408511.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-03T03_41_08.251099.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-03T04_09_00.933178.h5',
             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_04_30/rpg0/dl0/wf_runId_00310_configId_00001_2023-05-03T04_49_56.807340.h5',
#             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg3/dl0/wf_runId_00148_configId_00001_2023-03-22T08_28_41.911012.h5',
#             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg3/dl0/wf_runId_00148_configId_00001_2023-03-27T12_24_50.275131.h5',
#             '/data02/gammaflash/CIMONE/DL0/acquisizione_2023_03_22/rpg3/dl0/wf_runId_00148_configId_00001_2023-04-02T14_12_16.549322.h5'
]

def process_file_graph(filename,outdir=None):
    data = Reader().get_data_list(filename)
    t = np.arange(0,1000)
    for i in tqdm(range(len(data))):
        x = data[i][1000:2000]
        if not outdir:
            outdir = f'outdir/{filename.split('/')[-1]}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        test_bb_exp(t,x,ExponentialBlocks_Events,save = True,name=f'{outdir}/N_{sum(x)}_datanum_{i}.png')
        plt.clf()        
def process_file(filename):
    data = Reader().get_data_list(filename)
    t = np.arange(0,1000)
    unexp_list = []
    for i in tqdm(range(len(data))):
        x = data[i][1000:2000]
        n_blocks = len(bayesian_blocks(t,x,fitness=ExponentialBlocks_Events))-1      #identificare 3 blocchi è l'expected behaviour, se ne identifica un numero diverso vuol dire che bisogna investigare ulteriormente
        if n_blocks != 3:
            unexp_list.append(i)
    return [filename]+unexp_list
def process_file_index(unexp_row,mid_path):
    data = Reader().get_data_list(unexp_row[0])
    t = np.arange(0,1000)
    res = []
    for i in tqdm(unexp_row[1:]):        #careful, i is a str
        x = data[int(i)][1000:2000]
        outdir = f'outdir/{mid_path}/{unexp_row[0].split('/')[-1]}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        N = sum(x)
        test_bb_exp(t,x,ExponentialBlocks_Events,save=False,name=f'{outdir}/N_{N}_datanum_{i}.png')
        a = [y for y in data if (N-5000)<=sum(y[1000:2000])<=(N+5000)]                     #tutti i dati il cui N è vicino all'N di x
        b = [i for i in unexp_row[1:] if (N-5000)<=sum(data[int(i)][1000:2000])<=(N+5000)]      #tutti i dati unexp il cui N è vicino a x
        local_err_rate = len(b)/len(a)
        res.append([N,local_err_rate])
        plt.figtext(0.5,0.01,f'local_err_rate for N={N} is: {local_err_rate}')
        plt.savefig(f'{outdir}/N_{N}_datanum_{i}.png')
        plt.clf()
    return res

if __name__ == '__main__':
    with Pool(20) as p:
        if False:
            p.map(process_file_graph,filenames)
        if False:
            unexp_lists=p.map(process_file,filenames)
            print(unexp_lists)
            with open(f'outdir/unexp_beh_{time.time()}.csv','a+') as file:
                write = csv.writer(file)
                write.writerows(unexp_lists)
        if False:
            with open('outdir/unexp_beh_1707923326.0506213.csv') as file:
                csv_read = csv.reader(file)
                unexp_rows = list(csv_read)
            results = p.map(process_file_index,unexp_rows)
            for i in range(len(results)):
                res = np.array(results[i])
                plt.plot(res[:,0],res[:,1],'o')
                plt.figtext(0.5,0.01,f'local_err_rates for file {unexp_rows[i][0]}')
                plt.xlabel('N')
                plt.ylabel('loc_err_rate')
                tmp = unexp_rows[i][0].split('/')[-1]
                plt.savefig(f'outdir/loc_err_rate_{tmp}.png')
                plt.clf()
        if True:                         #combina le due funzioni sopra
                unexp_lists = p.map(process_file,filenames)
                tim = time.time()
                with open(f'outdir/unexp_beh_{tim}.csv','a+') as file:
                    write = csv.writer(file)
                    write.writerows(unexp_lists)
                with open(f'outdir/unexp_beh_{tim}.csv') as file:
                    csv_read = csv.reader(file)
                    unexp_rows = list(csv_read)
                results = p.starmap(process_file_index,zip(unexp_rows,repeat(tim)))
                for i in range(len(results)):
                    res = np.array(results[i])
                    try:
                        plt.plot(res[:,0],res[:,1],'o')
                        plt.figtext(0.5,0.01,f'local_err_rates for file {unexp_rows[i][0]}')
                        plt.xlabel('N')
                        plt.ylabel('loc_err_rate')
                        tmp = unexp_rows[i][0].split('/')[-1]
                        plt.savefig(f'outdir/{tim}/{tmp}/loc_err_rate_{tmp}.png')
                        plt.clf()
                    except Exception as e:
                        pass
