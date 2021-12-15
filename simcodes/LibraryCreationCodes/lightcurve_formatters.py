import numpy as np
import pandas as pd
import tqdm.notebook
def adapted_lightcurve_format(folder,source_id,suffix):
    return folder+f"{source_id}.csv"+suffix
class FormatLightcurves:
    def adapt_lighcurves(self,client=None,truncate:int=None,**kwargs):
        self.truncate_lightcurves=truncate
        if client is None:
            for i,row in tqdm.notebook.tqdm(self.adapted_dataset.iterrows(), 
                                            total=self.adapted_dataset.shape[0],
                                            desc='formatting lighcurves'):
                self.adapt_lightcurve_data(row,**kwargs)
        else:
            
            futures=client.map(self.adapt_lightcurve_data,
                               list(range(self.adapted_dataset.shape[0])))
            
            client.gather(futures)
        return self
    def get_adapted_lightcurve_name(self,source_id):
        return adapted_lightcurve_format(self.folder,source_id,self.lighcurve_suffix)
    def get_adapted_lightcurve(self,row):
        return pd.read_csv(self.get_adapted_lightcurve_name(row['source_id']),index_col=0)
        
    def get_unadapted_lightcurve(self,row):
        return  pd.read_csv(self.folder+f"{row['source_id']}.csv",index_col=0)
    def set_adapted_lightcurve(self,D,row):
        D.to_csv(self.folder+f"{row['source_id']}.csv"+self.lighcurve_suffix)
    def adapt_lightcurve_data(self,row:[pd.DataFrame,int],
                              period_percentage_interval=(0.1,10),
                              alternative_percentage_interval=(0.1,0.99)):
        if isinstance(row,int):
            row=self.get_adapted_dataset(row)
        D =  self.get_unadapted_lightcurve(row)
        for k,v in self.lightcurve_adaptation.items():
            D[v] =D[k]
        
        if self.check_lightcurve_data(D):
                
                D = D[D['catflags']==0][self.necessary_columns]
                if self.truncate_lightcurves is not None:
                    limit = min(D.shape[0],self.truncate_lightcurves)
                    D=D.sample(limit)
                D['t'] = D.t-D.t.min()
                rng=[period_percentage_interval[0]*row['input period'], period_percentage_interval[1]*row['input period'] ]
                Δt = D['t'].max()-D['t'].min()
                D['Range kind'] = 'period limited'
                if Δt <rng[1]:
                    D['Range kind'] = 'dt limited'
                    rng=[alternative_percentage_interval[0]*Δt,alternative_percentage_interval[1]*Δt]
                D['Range min'],D['Range max'] = rng
                
                self.set_adapted_lightcurve(D,row)
                return True
        return False
        
    def check_lightcurve_data(self,D):
        if 'Range kind' in D.columns:
            return  (D['Range kind']=='period limited').iloc[0]
        else:
            return np.sum(D['catflags']==0)>0 \
                and np.sum(D[D['catflags']==0].columns.isin(self.necessary_columns))==len(self.necessary_columns)
    def check_lightcurves(self):
        return sum(self.check_lightcurve_data(self.get_adapted_lightcurve(row)) 
                   for _,row in tqdm.notebook.tqdm(self.adapted_dataset.iterrows(),
                                                   total=self.adapted_dataset.shape[0],
                                                   desc='checking lighcurves'))