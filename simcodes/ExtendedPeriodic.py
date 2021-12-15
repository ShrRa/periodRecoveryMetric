import sys
sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/')
sys.path.append('/mnt/beegfs/scratch-noraid/ktisanic/Notebooks/PeriodFinding/simcodes/MC/')
import gatspy.periodic
import numpy as np
import pandas as pd
import scipy.stats
from dask.distributed import Client
#, SSHCluster
from .MC.df import save_locations, Chunk
import tqdm
class ExtendedLS(gatspy.periodic.LombScargleMultiband):
    def __init__(self,*args,**kwargs):
        gatspy.periodic.LombScargleMultiband.__init__(self,*args,**kwargs)
        self.t     = None
        self.dy    = None
        self.filts = None
        
    def copy_parameters(self,model:gatspy.periodic.LombScargleMultiband):
        self._best_period   = model._best_period
        self.unique_filts_  = model.unique_filts_
        self.ymean_by_filt_ = model.ymean_by_filt_
        self.omega          = 2 * np.pi / model._best_period
        self.theta          = model._best_params(self.omega)
        
    @staticmethod
    def get_parameters(model:gatspy.periodic.LombScargleMultiband)->dict:
        return dict(
                _best_period   = model._best_period,
                unique_filts_  = model.unique_filts_,
                ymean_by_filt_ = model.ymean_by_filt_,
                omega          = 2 * np.pi / model._best_period,
                theta          = model._best_params(2 * np.pi / model._best_period))
    
    @staticmethod
    def set_parameters(model:gatspy.periodic.LombScargleMultiband,parameters):
        model._best_period   = parameters['_best_period']
        model.unique_filts_  = parameters['unique_filts_']
        model.ymean_by_filt_ = parameters['ymean_by_filt_']

    def import_parameters(self,parameters:dict):
        self._best_period   = parameters['_best_period']
        self.unique_filts_  = parameters['unique_filts_']
        self.ymean_by_filt_ = parameters['ymean_by_filt_']
        self.omega          = parameters['omega']
        self.theta          = parameters['theta']
        
    def export_parameters(self)->dict:
        return dict(
                _best_period   = self._best_period,
                unique_filts_  = self.unique_filts_,
                ymean_by_filt_ = self.ymean_by_filt_,
                omega          = self.omega,
                theta          = self.theta)
        
    def predict(self, t:np.array,filts:np.array)->np.array:
        # need to make sure all unique filters are represented
        u, i = np.unique(np.concatenate([filts, self.unique_filts_]),
                         return_inverse=True)
        ymeans = self.ymean_by_filt_[i[:-len(self.unique_filts_)]]

        
        X = self._construct_X(self.omega, weighted=False, t=t, filts=filts)

        if self.center_data:
            return ymeans + np.dot(X, self.theta)
        else:
            return np.dot(X, self.theta)
def concatenating(g1:pd.DataFrame,folder:str,distribution_relative_error:scipy.stats.rv_continuous=None,
                 necessary_columns = ['t','mag','filt','magerr'],
                   adaptation={'filtercode':'filt',
                              'mjd':'t'},
                 copy_cols = ['Nterms']):
    Library = []

    for i, row in g1.iterrows():
        filename = folder+f"{row['source_id']}.csv"
        D =  pd.read_csv(filename,index_col=0)
        for k,v in adaptation.items():
                D[v] =D[k] 
        if np.sum(D['catflags']==0)>0 and np.sum(D[D['catflags']==0].columns.isin(necessary_columns))==len(necessary_columns):
               
                D = D[D['catflags']==0][necessary_columns]
                D['t'] = D.t-D.t.min()
                D['row'] = i
                D['relative_error']=D.magerr/D.mag
                for i in copy_cols:
                    D[i] = row[i]
                Library.append(D)
    lib= pd.concat(Library)
    if distribution_relative_error is not None:
        fit = distribution_relative_error.fit(lib.relative_error)
        return lib, fit, distribution_relative_error(*fit)
    else:
        return lib


def produce_samples(data:pd.DataFrame,Ns:np.array,distribution_relative_error:scipy.stats.rv_continuous):
    dfs = []
    num = 0
    for i,datum in data.iterrows():
        for N in Ns:
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=1)
            m.import_parameters(datum)
            phase = np.random.uniform(0,10,N)
            t     = np.tile(phase*datum['_best_period'],datum['unique_filts_'].size)
            t     = t-t.min()
            filt  = np.repeat(datum['unique_filts_'],phase.size)
            clean_prediction = m.predict(t,filt)
            relative_errs  = distribution_relative_error.rvs(size=t.size)
            errs = clean_prediction*relative_errs
            value = clean_prediction+np.random.normal(0,errs)
            dfs.append(pd.DataFrame({'t':t,'mag':value,'magerr':errs,'filt':filt,'data index':i,'simulation index':num,'_best_period':datum['_best_period'],
                                    'Size':N}))
            num+=1
    return pd.concat(dfs).reset_index(drop=True)
def produce_samples_dask(client,data:pd.DataFrame,Ns:np.array,distribution_relative_error:scipy.stats.rv_continuous,Nterms_base=1):
    def Wrapper_samples(i,data):
        datum = data.loc[i,:]
        ls=[]
        for j,N in enumerate(Ns):
            num=Ns.size*i+j
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms_base)
            m.import_parameters(datum)
            phase = np.random.uniform(0,10,N)
            t     = np.tile(phase*datum['_best_period'],datum['unique_filts_'].size)
            t     = t-t.min()
            filt  = np.repeat(datum['unique_filts_'],phase.size)
            clean_prediction = m.predict(t,filt)
            relative_errs  = distribution_relative_error.rvs(size=t.size)
            errs = clean_prediction*relative_errs
            value = clean_prediction+np.random.normal(0,errs)
            ls.append( pd.DataFrame({'t':t,'mag':value,'magerr':errs,'filt':filt,'data index':i,'simulation index':num,'_best_period':datum['_best_period'],
                                    'Size':N}))
        return ls
                #inputs = client.scatter(self.dfs)
                
    futures= client.map(Wrapper_samples,data.index,data=data)
    dfs = client.gather(futures)

    return pd.concat([_ for df in dfs for _ in df]).reset_index(drop=True)
def apply_format(Cepheids_dr3):
    for i in ['unique_filts_']:
        Cepheids_dr3[i] = Cepheids_dr3[i].apply(lambda x:eval(x.replace("' '","', '")) if isinstance(x,str) else np.array(x))
    for i in ['ymean_by_filt_','theta','omega','_best_period']:
        Cepheids_dr3[i] = Cepheids_dr3[i].apply(lambda x:np.array([float(y) for y in x.replace('[','').replace(']','').split()]) if isinstance(x,str) else x)
    return Cepheids_dr3
def chunk_list(x,N):
    L = len(x)//N
    return [x[N*i:N*(i+1)] for i in range(L)]+[x[L*N:]]
def produce_samples_dask_file(client,path,data:str,Ns:np.array,distribution_relative_error:scipy.stats.rv_continuous,Nterms_base=1,
                             chunksize=400):
    shape0 = pd.read_csv(data,index_col=0).shape[0]
    def Wrapper_samples(I,data,folder):
            i, j, N = I
            
            datum = apply_format(pd.read_csv(data,index_col=0)).iloc[i,:]

            num=Ns.size*i+j
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms_base)
            m.import_parameters(datum)
            phase = np.random.uniform(0,10,N)
            t     = np.tile(phase*datum['_best_period'],len(datum['unique_filts_']))
            t     = t-t.min()
            filt  = np.repeat(datum['unique_filts_'],phase.size)
            clean_prediction = m.predict(t,filt)
            relative_errs  = distribution_relative_error.rvs(size=t.size)
            errs = clean_prediction*relative_errs
            value = clean_prediction+np.random.normal(0,errs)
            df = pd.DataFrame({'t':t,'mag':value,'magerr':errs,'filt':filt,'data index':i,'simulation index':num,
                                    'Size':N})
            for key in ['_best_period','source_id', 'Expected', 'E-C', 'dof', 'χ2', 'χ2 dof', 'Type',
                       'Subtype', 'Nterms', 'LS period 0', 'LS period 0 score', 'χ2 0',
                       'χ2 0/dof', 'LS period 1', 'LS period 1 score', 'χ2 1', 'χ2 1/dof',
                       'LS period 2', 'LS period 2 score', 'χ2 2', 'χ2 2/dof', 'LS period 3',
                       'LS period 3 score', 'χ2 3', 'χ2 3/dof', 'LS period 4',
                       'LS period 4 score', 'χ2 4', 'χ2 4/dof', 'LS period 5',
                       'LS period 5 score', 'χ2 5', 'χ2 5/dof', 'LS period 6',
                       'LS period 6 score', 'χ2 6', 'χ2 6/dof', 'LS period 7',
                       'LS period 7 score', 'χ2 7', 'χ2 7/dof', 'LS period 8',
                       'LS period 8 score', 'χ2 8', 'χ2 8/dof', 'LS period 9',
                       'LS period 9 score', 'χ2 9', 'χ2 9/dof']:
                df[key] = datum[key]
            df.to_csv(folder+f'chunk_{num}.csv')
        
            return num,folder+f'chunk_{num}.csv'
                #inputs = client.scatter(self.dfs)
    Inputs0 = [[i,j,N] for i in range(shape0)  for j,N in enumerate(Ns)]
    Inputs = chunk_list(Inputs0,chunksize)
    
    dfs = []
    for inputs in tqdm.notebook.tqdm(Inputs):
        futures= client.map(Wrapper_samples,inputs,folder=path,data=data)
        dfs.extend(client.gather(futures))
    locations = dict(dfs)
    
    save_locations(path,locations)
    return Chunk(path, 'variable')

def LibraryCreation(g1:pd.DataFrame,library_name:str,folder:str,FourierComponents:list,
                   necessary_columns = ['t','mag','filt','magerr'],
                   adaptation={'filtercode':'filt',
                              'mjd':'t'},
                    row_adaptation={'Period any':'input period'}):
    Library = pd.DataFrame({})
    for k,v in row_adaptation.items():
                g1[v] =g1[k] 
    for i, row in g1.iterrows():
        for Nterms in FourierComponents:
            filename = folder+f"{row['source_id']}.csv"
            D =  pd.read_csv(filename,index_col=0)
            for k,v in adaptation.items():
                D[v] =D[k] 
            
            if np.sum(D['catflags']==0)>0 and np.sum(D[D['catflags']==0].columns.isin(necessary_columns))==len(necessary_columns):
                
                D = D[D['catflags']==0][necessary_columns]
                D['t'] = D.t-D.t.min()
                rng=[0.01*row['input period'], 100*row['input period'] ]
                Δt = D['t'].max()-D['t'].min()
                
                if Δt <rng[1]:
                    rng=[0.01*Δt,0.99*Δt]

                model = gatspy.periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                
                model.optimizer.period_range=tuple(rng)
                model.fit(D.t,D.mag,D.magerr, D.filt)        
                bestpers = model.find_best_periods(10,True)
                m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                m.import_parameters(m.get_parameters(model))
                m.copy_parameters(model)

                D['prediction'] = m.predict(D.t,D.filt)
                D['phase'] = (D.t/row['input period'])%1
                D['fit phase'] = (D.t/m._best_period)%1
                
                D.sort_values('phase',inplace=True)
                
                ϕ = np.linspace(0,1)
                t = ϕ*m._best_period
                filts = pd.unique(D.filt)
                predictions = pd.DataFrame({'t fit':np.tile( ϕ*m._best_period,filts.size),
                                            't Gaia':np.tile( ϕ*row['input period'],filts.size),
                                            'phase':np.tile(ϕ,filts.size),
                                           'filt':np.repeat(filts, ϕ.size)})
                
                    
                
                
                predictions['mag fit']  = m.predict(predictions['t fit'],predictions['filt'])
                predictions['mag Gaia'] = m.predict(predictions['t Gaia'],predictions['filt'])
                
                    
                params= m.export_parameters()
                params['source_id'] = row['source_id']
                params['Expected'] = row['input period']
                params['E-C'] = params['Expected']-params['_best_period']
                params['dof'] = D['prediction'].size-1
                params['χ2'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)
                params['χ2 dof'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)/params['dof']
                params['Type'] = row['Type']
                params['Subtype'] = row['Subtype']
                params['Nterms'] = Nterms
                
                for _i,(_p,score) in enumerate(zip(*bestpers)):
                    D[f'fit phase {_i}'] = (D.t/_p)%1
                    D[f'prediction {_i}'] =  model._predict(D.t,D.filt,_p)
                    
                    predictions[f't fit {_i}'] =np.tile( ϕ*_p,filts.size)
                    predictions[f'mag fit {_i}']  = model._predict(predictions[f't fit {_i}'],predictions['filt'],_p)
                    params[f"LS period {_i}"] = _p
                    params[f"LS period {_i} score"] = score
                    params[f'χ2 {_i}'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)
                    params[f'χ2 {_i}/dof'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)/(D.mag.size-1)
                Library = Library.append([params])

                #fig.savefig(f"plots/ztf/{i}_Nterms_{Nterms}_{row['source_id']}.png" ,bbox_extra_artists=[fig.suptitle(f"E-C={D['E-C']}")], bbox_inches='tight')
                #plt.close(fig)
                Library.reset_index(drop=True)
                Library.to_csv(library_name)
                o = np.logspace(*np.log10(model.optimizer.period_range),num=10000)
                
                yield Library, params,D, predictions,[o,model.periodogram(o)],bestpers#,#model.periodogram_auto()
def LibraryCreationdask(g1:pd.DataFrame,library_name:str,folder:str,FourierComponents:list,
                                               client,chunksize,
                        plotfolder=None,
                   necessary_columns = ['t','mag','filt','magerr'],
                   adaptation={'filtercode':'filt',
                              'mjd':'t'},
                    row_adaptation={'Period any':'input period'},
                       do_sort=None):
    
    for k,v in row_adaptation.items():
                g1[v] =g1[k] 
    g1.to_csv(library_name+'_input.csv')
    def LibraryDaskWrapper(inputs):
            i,Nterms = inputs
            
            row=pd.read_csv(library_name+'_input.csv',index_col=0).iloc[i]
            filename = folder+f"{row['source_id']}.csv"
            D =  pd.read_csv(filename,index_col=0)
            for k,v in adaptation.items():
                D[v] =D[k] 
            
            if np.sum(D['catflags']==0)>0 and np.sum(D[D['catflags']==0].columns.isin(necessary_columns))==len(necessary_columns):
                
                D = D[D['catflags']==0][necessary_columns]
                D['t'] = D.t-D.t.min()
                rng=[0.01*row['input period'], 100*row['input period'] ]
                Δt = D['t'].max()-D['t'].min()
                
                if Δt <rng[1]:
                    rng=[0.01*Δt,0.99*Δt]


                model = gatspy.periodic.LombScargleMultiband(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                
                model.optimizer.period_range=tuple(rng)
                model.fit(D.t,D.mag,D.magerr, D.filt)        
                #bestpers = model.find_best_periods(2,True)
                m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
                m.import_parameters(m.get_parameters(model))
                m.copy_parameters(model)

                #D['prediction'] = m.predict(D.t,D.filt)
                #D['phase'] = (D.t/row['input period'])%1
                #D['fit phase'] = (D.t/m._best_period)%1
                
                #D.sort_values('phase',inplace=True)
                
                #ϕ = np.linspace(0,1)
                #t = ϕ*m._best_period
                filts = pd.unique(D.filt)
                #predictions = pd.DataFrame({'t fit':np.tile( ϕ*m._best_period,filts.size),
                 #                           't Gaia':np.tile( ϕ*row['input period'],filts.size),
                 #                           'phase':np.tile(ϕ,filts.size),
                 #                          'filt':np.repeat(filts, ϕ.size)})
                
                    
                
                
                #predictions['mag fit']  = m.predict(predictions['t fit'],predictions['filt'])
                #predictions['mag Gaia'] = m.predict(predictions['t Gaia'],predictions['filt'])
                
                    
                params= m.export_parameters()
                params['source_id'] = row['source_id']
                params['Expected'] = row['input period']
                params['E-C'] = params['Expected']-params['_best_period']
                #params['dof'] = D['prediction'].size-1
                #params['χ2'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)
                #params['χ2 dof'] = np.sum((D['prediction']-D['mag'])**2/D['magerr']**2)/params['dof']
                params['Type'] = row['Type']
                params['Subtype'] = row['Subtype']
                params['Nterms'] = Nterms
                
                #for _i,(_p,score) in enumerate(zip(*bestpers)):
                #    D[f'fit phase {_i}'] = (D.t/_p)%1
                #    D[f'prediction {_i}'] =  model._predict(D.t,D.filt,_p)
                #    
                #    predictions[f't fit {_i}'] =np.tile( ϕ*_p,filts.size)
                #    predictions[f'mag fit {_i}']  = model._predict(predictions[f't fit {_i}'],predictions['filt'],_p)
                #    params[f"LS period {_i}"] = _p
                #    params[f"LS period {_i} score"] = score
                #    params[f'χ2 {_i}'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)
                #    params[f'χ2 {_i}/dof'] = np.sum((D[f'prediction {_i}']-D['mag'])**2/D['magerr']**2)/(D.mag.size-1)
              
                if plotfolder is not None:
                    o = np.logspace(*np.log10(model.optimizer.period_range),num=10000)
                    plot_LC(params,D, predictions,[o,model.periodogram(o)],bestpers,plotfolder)
                return params#,#model.periodogram_auto()
            else:
                return np.nan
    if do_sort is None:
        Inputs0 = [[i,Nterms]  for Nterms in FourierComponents for i in range(g1.shape[0])  ]
    else:
        Inputs0 = [[i,Nterms]  for Nterms in FourierComponents for i in g1.sort_values(do_sort).index  ]
    Inputs = chunk_list(Inputs0,chunksize)
    
    dfs = []
    for inputs in tqdm.notebook.tqdm(Inputs):
        futures= client.map(LibraryDaskWrapper,inputs)
        dfs.extend(client.gather(futures))
        pd.DataFrame(dfs).to_csv(library_name)

    dfs=pd.DataFrame(dfs)
    dfs.to_csv(library_name)
    return dfs

def plot_LC(datum,D,predictions,periodogram,best,folder):
        fig,ax = plt.subplots(3,2,figsize=(15,10))
        ax = ax.flatten()
        for i,[_filter,_lightcurve] in enumerate(D.groupby(D.filt)): 
            for a, phase in zip(ax,['phase','fit phase','fit phase 1', 'fit phase 2']):
                a.errorbar(x=_lightcurve[phase],
                           y=_lightcurve['mag'],
                           fmt='.',color=f'C{i}',
                           yerr=_lightcurve['magerr'],
                           label=_filter)


        for i,[_filter,_lightcurve] in enumerate(predictions.groupby(predictions.filt)): 
            for a, fit in zip(ax,['mag Gaia','mag fit','mag fit 1', 'mag fit 2']):
                a.plot(_lightcurve['phase'],_lightcurve[fit],color=f'C{i}',)
        for a, fit in zip(ax,['mag Gaia','mag fit','mag fit 1', 'mag fit 2']):
            a.invert_yaxis()

        ax[4].plot([i for i in range(best[0].size)],[datum[f'χ2 {i}/dof'] for i in range(best[0].size)])    
        ax[5].plot(*periodogram)
        ax[5].axvline(datum['Expected'],color='r',label='Gaia',lw=1)
        ax[5].axvline(datum['_best_period'],color='b',label='Fit',lw=1)
        for _i,_p in enumerate(best[0][1:4]):
            ax[5].axvline(datum[f'LS period {_i}'],color=f'C{_i}',label=f'Fit {_i}',lw=1)
        for a in ax[:4]:
            a.legend()
        ax[5].set_xscale('log')
        ax[4].set_yscale('log')
        ax[4].set_xlabel('n-th best fit')
        ax[4].set_ylabel(r'$\chi^2/(N-1)$')
        ax[5].set_xlabel('Period')
        ax[0].set_title('Gaia period %.4f'%datum['Expected'])
        ax[1].set_title('LS period %.4f'%datum['_best_period'])
        ax[2].set_title('LS period from 2nd best fit %.4f'%datum['LS period 1'])
        ax[3].set_title('LS period from 3rd best fit %.4f'%datum['LS period 2'])
        fig.savefig(folder\
                    +simcodes.formatters.plot_name(datum['Subtype'],ID=datum['source_id'],Nterms=datum['Nterms']),
                    bbox_extra_artists=[fig.suptitle(f"ID={datum['source_id']} {datum['Subtype']}")], 
                    bbox_inches='tight')
        plt.close(fig)
