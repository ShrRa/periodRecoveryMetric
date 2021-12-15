#import sys
#sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/')
#sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/MC/')

import numpy as np
import pandas as pd
import scipy.stats
from dask.distributed import Client, SSHCluster
import tqdm.notebook
from .LibraryCreationCodes import FormatLightcurves,FormatDataset
from .fitters import fit_lightcurve,needed_row_columns
#from .fitters import fit_lightcurve_input_tester as fit_lightcurve
class LibraryCreation(FormatLightcurves,FormatDataset):
    necessary_columns = ['t','mag','filt','magerr']
    lightcurve_adaptation        = {'filtercode':'filt',
                              'mjd':'t'}
    row_adaptation    = {'Period any':'input period'}
    library_suffix =    '_input.csv'        
    lighcurve_suffix='_cropped.csv'
    def __init__(self,
                 df:pd.DataFrame,
                 library_name:str,
                 folder:str,):
        FormatLightcurves.__init__(self)
        FormatDataset.__init__(self,library_name,df)
        self.folder=folder
    
    
    def create_library(self,client):
        dfs = []
        for inputs in tqdm.notebook.tqdm(self.Inputs,desc='Creating library'):
            futures= client.map(fit_lightcurve,inputs,adapted_dataset=self.adapted_dataset_name)
            dfs.extend(client.gather(futures))
            self.resulting_library=pd.DataFrame(dfs)
            self.resulting_library.to_csv(self.library_name)

        self.resulting_library=pd.DataFrame(dfs)
        self.resulting_library.to_csv(self.library_name)
        return self
    def make_inputs(self,chunksize,FourierComponents):
        Inputs0 = [[i,Nterms,row[needed_row_columns]]  for Nterms in FourierComponents for i,row in self.adapted_dataset.iterrows()]
        self.Inputs = self.chunk_list(Inputs0,chunksize)
        return self
    def chunk_list(self,x,N):
        L = len(x)//N
        return [x[N*i:N*(i+1)] for i in range(L)]+[x[L*N:]]