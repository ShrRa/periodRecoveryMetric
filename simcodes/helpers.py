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
def p16(x):
    
    return np.percentile(x,16)
def p84(x):
    return np.percentile(x,84)
def med_sig(x):
    return (np.percentile(x,75)-np.percentile(x,25))/1.349
def unexplained_variance(x):
    return np.sum((x-np.median(x))**2)/x.size-med_sig(x)**2
def Outliers1(x):
    N = x.size
    s = med_sig(x)
    m = np.median(x)
    return (np.sum(x<m-s)+ np.sum(x>m+s))/N
def Outliers3(x):
    N = x.size
    s = med_sig(x)
    m = np.median(x)
    return (np.sum(x<m-3*s)+ np.sum(x>m+3*s))/N
def plot_lightcurve(lightcurve_p,Kbf,data):    
    bins = pd.unique(lightcurve_p.index.get_level_values(1))[::5]
    fig,ax = plt.subplots(1,bins.size,figsize=(5*bins.size,5))
    ax = np.reshape(ax,-1)
    for TYPE in pd.unique(pd.unique(lightcurve_p.index.get_level_values(0))):
        for i,N in enumerate(bins):
            for f in pd.unique(lightcurve_p.index.get_level_values(2)):
                m = lightcurve_p.loc[(TYPE,N,f)]
                _=ax[i].plot(m.index,m['p_50'])[0].get_color()
                cut = Kbf[4]==f
                #print(cut.shape,Kbf[5].shape,Kbf[6].shape)
                ax[i].plot(Kbf[6],Kbf[5][cut][0],color=_,ls='dashed')
                ax[i].fill_between(m.index,m['p_16'],m['p_84'],alpha=0.5,color=_,label=f)
                c = data[data.filt==f]
                ax[i].plot(c.phase, c.mag,'.',color=_)
            ax[i].legend()
            ax[i].set_xlabel('Phase')
            ax[i].set_ylabel('Magnitude')
            ax[i].set_title(N)
            ax[i].invert_yaxis()
    fig.tight_layout()
    return fig
def compute_stats(simulated_periods,chosen_col,groupby_col):
    import scipy.stats as sc
    moments = {chosen_col: [
                                                                                lambda x: np.percentile(x.dropna(),16),
                                                                                lambda x: np.percentile(x.dropna(),50),
                                                                                lambda x: np.percentile(x.dropna(),84),
                                                                                lambda x: np.max(x.dropna()),
                                                                                lambda x: np.min(x.dropna()),
                                                                                lambda x: sc.sem(x.dropna()),
                                                                                lambda x: sc.skew(x.dropna()),
                                                                                lambda x: np.std(x.dropna())]}
    moments = simulated_periods.groupby(simulated_periods[groupby_col],).agg(moments)
    moments.columns = ['p_16', 'p_50', 'p_84','max','min','sem','skewness','std']
    return moments
def periods(simulated_periods, P0,label,FIG=None):
    import scipy.stats as sc
    mean = simulated_periods.groupby(simulated_periods['N'],).median()
    p16 =  simulated_periods.groupby(simulated_periods['N'],).agg(lambda x:np.percentile(x.dropna(),16))
    p84 =  simulated_periods.groupby(simulated_periods['N'],).agg(lambda x:np.percentile(x.dropna(),84))
    p100 =  simulated_periods.groupby(simulated_periods['N'],).agg(lambda x:np.max(x.dropna(),))
    skewness  =  simulated_periods.groupby(simulated_periods['N'],).agg(lambda x:sc.skew(x.dropna(),))
    std = simulated_periods.groupby(simulated_periods['N'],).agg(lambda x:np.std(x.dropna(),))
    moments = {'P': [
                                                                                lambda x: np.percentile(x.dropna(),16),
                                                                                lambda x: np.percentile(x.dropna(),50),
                                                                                lambda x: np.percentile(x.dropna(),84),
                                                                                lambda x: np.max(x.dropna()),
                                                                                lambda x: np.min(x.dropna()),
                                                                                lambda x: sc.sem(x.dropna()),
                                                                                lambda x: sc.skew(x.dropna()),
                                                                                lambda x: np.std(x.dropna())]}
    moments = simulated_periods.groupby(simulated_periods['N'],).agg(moments)
    moments.columns = ['p_16', 'p_50', 'p_84','max','min','sem','skewness','std']
    display(moments)
    if FIG is None:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
    else:
        fig,ax=FIG
    ax[0].plot(moments.index,moments.p_50)
    ax[0].plot(moments.index,moments['max'])
    ax[0].set_ylim(3,3.5)
    ax[0].fill_between(moments.index,moments.p_16,moments.p_84,alpha=0.5,color='C0')
    ax[0].axhline(P0,color='k',ls='dashed')
    
    ax[0].set_ylabel('Period [d]')
    ax[1].plot(moments.index,moments.skewness,label=label)
    
    ax[2].plot(moments.index,moments['std'],label=label)
    x = np.linspace(moments.index.min(),moments.index.max())
    xm = moments[moments['std']==moments['std'].max()].index.values[0]
    ax[2].set_xscale('log')
    ax[2].plot(x,moments['std'].max()*np.sqrt(xm/x),'k',ls='dashed')
    ax[2].set_yscale('log')
    ax[2].set_ylabel('std')
    ax[1].set_ylabel('Skewness')
    for axx in ax:
        axx.set_xlim(10,100)
        axx.legend()
        axx.set_xlabel('Number of measurements per band')
    return fig,ax