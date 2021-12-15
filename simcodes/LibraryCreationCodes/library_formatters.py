import numpy as np
import pandas as pd
import tqdm.notebook

class FormatDataset:    
    def __init__(self,library_name,df):
        self.library_name=library_name
        if isinstance(df,str):
            self.adapted_dataset_name=self.library_name+self.library_suffix
            self.adapted_dataset=pd.read_csv(self.adapted_dataset_name,index_col=0)
        else:
            self.adapted_dataset_name=self.library_name+self.library_suffix
            df.to_csv(self.adapted_dataset_name)
            self.adapted_dataset= df
    def adapt_dataset(self,do_sort=None):
        self.adapted_dataset.rename(columns=self.row_adaptation)
        if do_sort is not None:
            self.adapted_dataset=self.adapted_dataset.sort_values(do_sort)
        self.adapted_dataset.to_csv(self.adapted_dataset_name)
        
        return self
    
    def get_adapted_dataset(self,i):
        return pd.read_csv(self.adapted_dataset_name,index_col=0).iloc[i]
    def truncate_dataset(self):
        self.adapted_dataset.loc[:,'Keep']=[self.check_lightcurve_data(self.get_adapted_lightcurve(row)) 
                                      for i,row in tqdm.notebook.tqdm(self.adapted_dataset.iterrows(),
                                                                     total=self.adapted_dataset.shape[0],
                                                   desc='Truncating dataset')]
        self.adapted_dataset = self.adapted_dataset[self.adapted_dataset['Keep']==True].reset_index(drop=True)
        self.adapted_dataset.to_csv(self.adapted_dataset_name)
        return self
    def write_lightcurves_to_dataset(self):
        self.adapted_dataset['lightcurve name'] = [self.get_adapted_lightcurve_name(row['source_id']) 
                                                 for _, row in tqdm.notebook.tqdm(self.adapted_dataset.iterrows(),
                                                                                  total=self.adapted_dataset.shape[0],
                                                   desc='writing lightcurve locations')]
        self.adapted_dataset.to_csv(self.adapted_dataset_name)
        
        return self