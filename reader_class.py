import tables
import os
from astropy.stats import bayesian_blocks
from bb_exponential import ExponentialBlocks_Events
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat
import numpy as np

class Reader:
    def get_data_list(self,filename,startIndex=0,endIndex=-1):
        '''get data from h5 file starting by startIndex and ending at endIndex
        '''
        data_list = []
        with tables.open_file(filename,mode='r') as h5file:
            group = h5file.get_node('/waveforms')
            if endIndex == -1:
                endIndex = group._g_getnchildren()
            for i,data in enumerate(group):
                if i >= startIndex:
                    arr = data[:,-1]
                    data_list.append(arr)
                if i>=endIndex-1:
                    break
            return data_list
    def _make_graph(self,t,x,actual_outdir,name):
        plt.step(t, x)
        flag = False
        N = sum(x)
        ncp_prior = ExponentialBlocks_Events().p0_prior(N)
        while not flag:  #se ci sono errori di tipo 'troppi blocchi' faccio un ricalcolo usando un ncp_prior più alto
            xcoords = bayesian_blocks(t,x,fitness=ExponentialBlocks_Events,ncp_prior=ncp_prior)
            blocks_lengths = xcoords[1:]-xcoords[:-1]    
            if np.count_nonzero(blocks_lengths == 2)>2: #se ci sono più blocchi di lunghezza 2, allora c'è un errore di tipo 'troppi blocchi'
                ncp_prior = ncp_prior + 100
            else:
                flag = True
        for xc in xcoords:
            plt.axvline(x=xc, color='grey', alpha=0.3)
        for i in range(len(xcoords)-1):
            edge_l = xcoords[i]
            edge_r = xcoords[i+1]
            params = ExponentialBlocks_Events(ncp_fun=(lambda N: ExponentialBlocks_Events().p0_prior(N)-100)).get_parameters(edge_l,edge_r,t,x)
            a = params['a']
            gamma = params['gamma']
            plot_t = np.linspace(edge_l,edge_r,1000)
            plot_x = gamma * np.exp(a*(plot_t - edge_r))
            plt.plot(plot_t,plot_x)
        plt.figtext(0.5,0.01,f'subgraph area: {sum(x)}')
        plt.savefig(actual_outdir+'/'+name)
        plt.clf()
    def process_file(self,filename,startIndex=0,endIndex=-1,outdir='outdir',num_workers=1):
        '''
        filename: str, path to a h5 file containing binned data
        startIndex: int, starting index of data to process
        endIndex: int, index of last data to process
        outdir: str, path of the folder where a folder named like filename will be created to contain
                     the graphs of processed data
        num_workes: int, number of threads to use to speed up processing time
        
        processes the file via bb_exponential,returns nothing and as a side effect creates a folder 
        with the same name as the filename inside outdir with all the graphs of the processed data
        '''
        actual_outdir=os.path.join(outdir,filename.split('/')[-1])
        if not os.path.exists(actual_outdir):
            os.makedirs(actual_outdir)
        data = self.get_data_list(filename,startIndex=startIndex,endIndex=endIndex)
        if endIndex == -1:
            endIndex = startIndex+len(data)
        t = np.arange(0,1000)
        x_s = [x[1000:2000] for x in data]
        zipped_data = zip(repeat(t),x_s,repeat(actual_outdir),[str(i)+'.png' for i in np.arange(startIndex,endIndex)])
        with Pool(num_workers) as p:
            p.starmap(self._make_graph,zipped_data)