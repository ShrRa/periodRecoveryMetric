#import sys
#sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/')
#sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/MC/')
import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from gatspy import datasets, periodic
import astropy.timeseries as astropy_timeseries
from astropy.table import Table, vstack
import warnings
from scipy.stats import sem
from dask.distributed import Client, SSHCluster
import dask

class correctedNaiveMultiband:
    def __init__(self,*args,**kwargs):
        
        self.args = args
        self.kwargs = kwargs
    def fit(self, T, M, ME, F):
        self.filters = pd.unique(F)
        self.models = {}
        for _filter in self.filters:
            cut = F==_filter
            self.models[_filter]=periodic.LombScargleFast(*self.args,**self.kwargs)
            self.models[_filter].optimizer.period_range=(0.1, 10)
            self.models[_filter].fit(T[cut],M[cut],ME[cut])
    def predict(self, T,filts):
        predictions = []
        for i,_filter in enumerate(pd.unique(filts[:,0])):
           
            predictions.append(self.models[_filter].predict(T[:,i]))
        return np.array(predictions)
    @property
    def best_period(self):
        return np.array([self.models[_filter].best_period for _filter in self.filters])
    def periodogram_auto(self):
        PA = [self.models[_filter].periodogram_auto() for _filter in self.filters]
        return np.array([p[0] for p in PA]), np.array([p[1] for p in PA])
    def periodogram(self,*args,**kwargs):
        return np.array([self.models[_filter].periodogram(*args,**kwargs) for _filter in self.filters])
def make_samples(df,filtercol,Number_in_simulation):
            DF = pd.DataFrame()
            for _filter in pd.unique(df[filtercol]):
                cut = df[df[filtercol]==_filter]
                
                DF = DF.append(cut.iloc[np.random.randint(0,cut.shape[0],min(Number_in_simulation,cut.shape[0])),:])
            return DF.reset_index()
def make_index_samples(df,filtercol,Number_in_simulation):
            DF = pd.DataFrame()
            for _filter in pd.unique(df[filtercol]):
                cut = df[df[filtercol]==_filter]
                
                DF = DF.append(cut.iloc[np.random.randint(0,cut.shape[0],min(Number_in_simulation,cut.shape[0])),:])
            return DF.index
def testing(Number_in_simulation, P0, original_file, TYPE='fast',  Periodogram_auto=False, phase_plot=True):

        o = np.linspace(1/100,24,10000)
        def process(df, filters=list('IV')):

            if TYPE == 'fast':
                model = periodic.LombScargleMultibandFast(fit_period=True, Nterms=1,optimizer_kwds=dict(quiet=True))
                
                model.optimizer.period_range=(0.1, 10)
            if TYPE == 'slow':
                model = periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True))
                
                model.optimizer.period_range=(0.1, 10)
            if TYPE == 'naive':
                #model = periodic.NaiveMultiband(fit_period=True)
                model = correctedNaiveMultiband(fit_period=True,optimizer_kwds=dict(quiet=True))
            
            model.fit(df['t'], df['mag'], df['magerr'], df['filt'])
            if TYPE == 'naive':
                    best_period = np.mean(model.best_period)
            else:
                    best_period = model.best_period
            if phase_plot:

                tfit = np.linspace(0, model.best_period, 1000)
                filtsfit = np.array(filters)[:, np.newaxis]
                magfit = model.predict(tfit, filts=filtsfit)

                if np.size(model.best_period)>1:
                    phase = (df['t'][np.newaxis,:] / model.best_period[:,np.newaxis]) % 1
                    phasefit = (tfit.T / model.best_period[:,np.newaxis])
                else:
                    phase = (df['t'] / model.best_period) % 1
                    phasefit = (tfit / model.best_period)
                
                
                if Periodogram_auto:
                        return best_period, tfit, filtsfit, magfit, phasefit,model.periodogram_auto()
                return best_period, tfit, filtsfit, magfit, phasefit,[o,model.periodogram(o)]
                
            return best_period
        if isinstance(original_file,str):
            df = pd.read_csv(original_file)
        else:
            df = original_file
        
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                K = process(df, np.unique(df["filt"]))
        if phase_plot:
            return  Number_in_simulation,K[0],\
                    K[-1][0],\
                    K[-1][1],\
                    K[2].flatten(), K[3], K[4],TYPE   
        else:
            return Number_in_simulation,K,TYPE  
def unpack(X):
    N = len(X)
    L = len(X[0])
    outs = [np.array([X[i][j] for i in range(N)]) for j in range(L)]
    return outs
def window(X:pd.DataFrame, c:np.float,w:np.float,phasecol='phase'):
    phases=X['phase']
    a = c-w/2.
    b = c+w/2.
    return (phases<a) | (phases>b)
class MCSimulation:
    memory_types = ['df', 'index']
    def __init__(self, data_, P0, initial_lightcurve,verbose=True,memory_type='df'):
        self.P0                 = P0
        self.data_              = data_
        self.initial_lightcurve = initial_lightcurve
        
        self.simulations  = {}
        self.lightcurve_p = {}
        self.lightcurve = {}
        self.simulated_periods = {}
        self.bootstrap_samples  = {}
        self.verbose = verbose
        self.memory_type = memory_type
    @property
    def memory_type(self):
        return self._memory_type
    @memory_type.setter
    def memory_type(self,x):
        if x not in self.memory_types:
            raise ValueError('Invalid memory type')
        self._memory_type = x
    def do_best(self,):
        self.best_fitting       = testing(None, self.P0,self.initial_lightcurve,TYPE="fast",Periodogram_auto=True,phase_plot=True)
    def compute_phase(self):
        df = pd.read_csv(self.initial_lightcurve)
        df.loc[:,'phase'] =(df['t'] / self.P0) % 1
        df.to_csv(self.initial_lightcurve)
    def produce_bootstrap(self,Sizes,Nreps):
        if self.memory_type == 'df':
            sims = pd.read_csv(self.initial_lightcurve)
            self.dfs = [[i,make_samples(sims,'filt', i)] for i in np.sort(np.tile( Sizes, Nreps))]
        if self.memory_type== 'index':
            self.sims = pd.read_csv(self.initial_lightcurve)
            self.dfs = [[i,make_index_samples(self.sims,'filt', i)] for i in np.sort(np.tile( Sizes, Nreps))]
    def remove_window(self,*args,**kwargs):
        self.dfs = [[i, K[window(K,*args,**kwargs)]] for i, K in self.dfs]
    def remove_window_variable_width(self,centers_widths,*args,**kwargs):
        if self.memory_type =='df':
            self.dfs = [[i, K[window(K,centers_widths[j][0],centers_widths[j][1],*args,**kwargs)]] for j in range(len(centers_widths)) for i, K in self.dfs ]
            self.dfs = [[K.shape[0], K] for i, K in self.dfs ]
        if self.memory_type== 'index':
            self.dfs = [[i, window(self.sims.iloc[K,:],centers_widths[j][0],centers_widths[j][1],*args,**kwargs).index] for j in range(len(centers_widths)) for i, K in self.dfs ]
            self.dfs = [[np.size(K), K] for i, K in self.dfs ]
        
    def run_simulation(self,method, cluster=None,output=None):
        def MapWrapper(i):
            if self.memory_type== 'index':
                return np.array(testing(i[0],self.P0,self.sims.loc[i[1],:].reset_index(),method, False,phase_plot = self.verbose))
            if self.memory_type== 'df':
                return np.array(testing(i[0],self.P0,i[1],method, False,phase_plot = self.verbose))
        
        
        if cluster is None:
            L= [*map(MapWrapper,self.dfs)]
        else:
            with Client(cluster) as client:
                #inputs = client.scatter(self.dfs)
                
                futures= client.map(MapWrapper,self.dfs)
                L = client.gather(futures)
                
        print(f"D",end='\t')
        self.simulations[method] = unpack(L)           
        self.bootstrap_samples[method] = []
        for i in range(len(self.dfs)):
            if self.memory_type=='df':
                self.bootstrap_samples[method].append(self.dfs[i][1])
            if self.memory_type=='index':
                self.bootstrap_samples[method].append(self.sims.loc[self.dfs[i][1],:].reset_index())
            self.bootstrap_samples[method][-1]['run']=i
        self.bootstrap_samples[method] = pd.concat(self.bootstrap_samples[method])
        if self.verbose:        
            self.lightcurve_p[method],\
            self.lightcurve[method],\
            self.simulated_periods[method] = self.decompose(self.simulations[method],method)
        else:
            self.simulated_periods[method] = self.decompose(self.simulations[method],method)
        if output is not None:
            self.save_simulation(output)
    
    def decompose(self,VectorizedInput,method):
        if self.verbose:
            MCN,MCPeriods,\
            MCPeriodogram_p, MCPeriodogram_A,\
            MCFilter, MCMag,  MCPhase,MCType = VectorizedInput
        
            simulated_periodogram = pd.DataFrame({"N":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                                 "P":np.tile(MCN,MCPeriodogram_p.shape[1]).reshape(-1,MCPeriods.size).T.flatten(),
                                                 "o":MCPeriodogram_p.flatten(),
                                                 'A':MCPeriodogram_A.flatten()
                                                 })
            self.simulated_periods[method]= pd.DataFrame({"N":MCN,"P":MCPeriods})
            lightcurve = pd.DataFrame()
            for _filter in pd.unique(MCFilter.flatten()):

                cut = MCFilter==_filter
                phase =  MCPhase
                for i,N in enumerate(pd.unique(MCN)):

                    lightcurve=lightcurve.append(pd.DataFrame({"N":MCPhase.shape[-1]*[N], 
                                                               "Filter":MCPhase.shape[-1]*[_filter], 
                                                               "phase":MCPhase[i], 
                                                               "mag":MCMag[cut][i],"kind":MCType[i]}),ignore_index=True)
            lightcurve['phase bin'] = pd.cut(lightcurve.phase,20).apply(lambda x: x.mid)
            self.lightcurve[method] = lightcurve
            self.lightcurve_p[method] = lightcurve.groupby(["kind",'N','Filter','phase bin',]).agg({'mag': [
                                                                                    lambda x: np.percentile(x,16),
                                                                                    lambda x: np.percentile(x,50),
                                                                                    lambda x: np.percentile(x,84),
                                                                                    lambda x: np.max(x),
                                                                                    lambda x: np.min(x),
                                                                                    lambda x: sem(x)]})
            self.lightcurve_p[method].columns = ['p_16', 'p_50', 'p_84','max','min','sem']
            self.lightcurve_p[method]['method'] = method
            self.lightcurve[method]['method'] = method
            self.simulated_periods[method]['method'] = method
            self.bootstrap_samples[method]['method'] = method
            self.Lightcurve = pd.concat(self.lightcurve.values())
            self.Lightcurve_p = pd.concat(self.lightcurve_p.values())
            self.Simulated_periods = pd.concat( self.simulated_periods.values())
            self.Bootstrap_samples = pd.concat(self.bootstrap_samples.values())
            return self.lightcurve_p[method], self.lightcurve[method],self.simulated_periods[method]
        else:
            MCN,MCPeriods,MCType = VectorizedInput
            self.simulated_periods[method]= pd.DataFrame({"N":MCN,"P":MCPeriods.astype(float)})
            self.bootstrap_samples[method]['method'] = method
            self.Simulated_periods = pd.concat( self.simulated_periods.values())
            self.Bootstrap_samples = pd.concat(self.bootstrap_samples.values())
            return self.simulated_periods[method]
    def save_simulations(self,output_):
        if self.verbose:
            self.Lightcurve.to_csv(output_+'_lightcurve.csv')
            self.Lightcurve_p.to_csv(output_+'_lightcurve_p.csv')
        self.Simulated_periods.to_csv(output_+'_periods.csv')
        self.Bootstrap_samples.to_csv(output_+'_samples.csv')
        
    def load_simulations(self,output_):
        if self.verbose:
            self.Lightcurve        = pd.read_csv(output_+'_lightcurve.csv')
            self.Lightcurve_p      = pd.read_csv(output_+'_lightcurve_p.csv')
            self.lightcurve        = {method: self.Lightcurve[ self.Lightcurve['method']==method] for method in pd.unique( self.Lightcurve['method'])} 
            self.lightcurve_p      = {method: self.Lightcurve_p[ self.Lightcurve_p['method']==method] for method in pd.unique( self.Lightcurve_p['method'])}
            self.lightcurve_p      = {method:df.set_index(["kind",'N','Filter','phase bin',]) for method,df in self.lightcurve_p.items()}
        self.Simulated_periods = pd.read_csv(output_+'_periods.csv')
        self.Bootstrap_samples = pd.read_csv(output_+'_samples.csv')
        self.simulated_periods = {method: self.Simulated_periods[ self.Simulated_periods['method']==method] for method in pd.unique( self.Simulated_periods['method'])}
        self.bootstrap_samples = {method: self.Bootstrap_samples[ self.Bootstrap_samples['method']==method] for method in pd.unique( self.Bootstrap_samples['method'])}      
        