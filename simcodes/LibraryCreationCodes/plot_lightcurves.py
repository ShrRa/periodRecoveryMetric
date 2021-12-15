import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .. import fitters,formatters
from fitters import ExtendedLS

def compute_stats(df,by):
        def p16(x):
            return np.percentile(x,16)
        def p84(x):
            return np.percentile(x,84)
        def σ(x):
            return 0.5*(np.percentile(x,84)-np.percentile(x,16))
        def σl(x):
            return (np.percentile(x,50)-np.percentile(x,16))
        def σu(x):
            return (np.percentile(x,84)-np.percentile(x,50))
        return df.groupby(df[by]).agg(['median',p16,p84,σ,σl,σu]).dropna()
def compute_predictions(inputs,lc_folder):
    source_id,Nterms,row=inputs 
    D=pd.read_csv(lc_folder+str(source_id)+'.csv_cropped.csv',index_col=0)
    ext=ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
    ext.import_parameters(row)
    D['predictions']=ext.predict(D.t,D.filt)
    D['phase'] =  (D.t/row['_best_period'])%1
    D['Z-score'] = (D['predictions']-D['mag'])/D['magerr']
    ϕ = np.linspace(0,2,100)
    
    filts = pd.unique(D.filt)
    ϕ,filts=np.tile(ϕ,filts.size),np.repeat(filts, ϕ.size)
    t=ϕ*ext._best_period
    mag = ext.predict(t,filts)
    #D.to_csv(folders.ZTF_data+'3_arcsec/'+str(source_id)+f'.csv_cropped_Nterms_{Nterms}.csv')
    return D,pd.DataFrame({'phase':ϕ,'predictions':mag,'filt':filts})    

def plot(source_id,rows,folder,lc_folder):
        fig,ax = plt.subplots(2,2,figsize=(10,10),subplot_kw={'xlabel':'Phase'})
        ax=ax.flatten()
        residual_ax = ax[-1]
        ax = ax[:-1]
        for j, (a,(Nterms,row)) in enumerate(zip(ax,rows.groupby('Nterms'))):

            row = row.iloc[0]
            D,continuous_pred=compute_predictions([source_id,Nterms,row],lc_folder)
            D1 = D.copy()
            D1['phase']=D1['phase']+1
            D = D.append(D1)
            D['phase_bins'] = pd.cut(D['phase'],np.linspace(0,2,10))
            group = compute_stats(D,'phase_bins')
            for i,[_filter,_lightcurve] in enumerate(D.groupby(D.filt)): 
                    _lightcurve = _lightcurve.sort_values('phase')

                    a.errorbar(x=_lightcurve['phase'],
                                           y=_lightcurve['mag'],
                                           fmt=',',color=f'C{i}',
                                           yerr=_lightcurve['magerr'],
                               elinewidth=0.1
                                           )
            for i,[_filter,_lightcurve] in enumerate(continuous_pred.groupby(continuous_pred.filt)): 
                    _lightcurve = _lightcurve.sort_values('phase')


                    a.plot(_lightcurve['phase'],
                                       _lightcurve['predictions'],
                                       color=f'C{i}',

                                       label=_filter,
                           lw=2
                                      )

            a.set_xlim(0,2)
            a.invert_yaxis()
            a.legend()
            a.set_ylabel('$m$')
            a.set_title(f'{Nterms} Fourier terms')

            residual_ax.plot(D['phase'],D['Z-score'],f'C{j},')
            residual_ax.plot(group['phase']['median'],group['Z-score']['median'],f'C{j}',label=Nterms,lw=2)
            residual_ax.fill_between(group['phase']['median'],group['Z-score']['p16'],group['Z-score']['p84'],color=f'C{j}',alpha=0.5)
            residual_ax.legend(title='F. terms')
            residual_ax.set_xlim(0,2)
            residual_ax.set_ylim(-5,5)
            residual_ax.set_ylabel('Z-score')
            residual_ax.axhline(0,color='k',ls='dashed')
            residual_ax.axhline(1,color='k',ls='dotted')
            residual_ax.axhline(-1,color='k',ls='dotted')

        fig.savefig(folder\
                        +formatters.plot_name(row['Type'],row['Subtype'],ID=source_id),
                        bbox_extra_artists=[fig.suptitle(f"ID={source_id} {row['Subtype']}")], 
                        bbox_inches='tight')
        plt.close(fig)