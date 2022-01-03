import rubin_sim.maf as maf
import rubin_sim.utils as rsUtils
import numpy as np

__all__ = ["KuiperVS"]

class KuiperVS(maf.BaseMetric):
    """Measure the uniformity of distribution of observations across phased light curve using Kuiper test. 
    0 means perfectly uniform distribution, 1 means delta-function. For small number of observations (<~30, in reality depends
    on the shape of the light curve) this metric becomes unreliable.
     
    Parameters
    ----------
    mjdCol : string
        name of the column containing the starting date of observation
    period : float 
        period for which we want to check the uniformity of the phase coverage
    filters : list-like
        list of filters used for calculating Kuiper value
    amplitudes : float or list-like
        list of amplitudes of the stellar variablility (mags) of the same dimensionality as 'filters'
    starMags : float 
        list of mean magnitudes of the stars (mags) of the same dimensionality as 'filters'
    
    Returns
    -------
    Kuiper value in range [0;1] where 0 means perfectly uniform distribution and 1 means delta-function
    """
    
    def __init__(self, mjdCol='observationStartMJD', period=1.,filters=['u','g','r','i','z','y'],
                 amplitudes=[0.5]*6,starMags=[23.5]*6,**kwargs):
        
        self.mjdCol = mjdCol
        self.filterCol='filter'
        self.fiveSigmaDepthCol='fiveSigmaDepth'
        self.filters=list(filters)
        self.amplitudes=list(amplitudes)
        self.starMags=list(starMags)
        self.badval=-666
        super(KuiperVS, self).__init__(col=[self.mjdCol,self.filterCol,self.fiveSigmaDepthCol], 
                                       units='Kuiper value, 0-1',badval=self.badval, **kwargs)
        self.period=period

    def run(self, dataSlice, slicePoint=None):
        if len(self.filters)!=len(self.amplitudes) or len(self.filters)!=len(self.starMags):
            raise RuntimeError("Filters, amplitudes and starMags should be of the same length")
        
        # Selecting only those observations that fall within magnitude limits 
        ind = [] 
        for filt,mag,amp in zip(self.filters,self.starMags,self.amplitudes):
            ind.extend(np.where((dataSlice[self.filterCol]==filt) & (dataSlice[self.fiveSigmaDepthCol]>(mag+amp)))[0])
        if ind == []:
            return self.badval
        dSlice = dataSlice[np.array(ind)]
        # How to implement limiting for brightness clipping, i.e. for situations when 
        # the source is too bright to be observed?
        
        # If only one observation, it's delta function and DKuiper=1
        if dSlice[self.mjdCol].size == 1:
            return 1
        
        # Create phased cadence
        phased=np.sort((dSlice[self.mjdCol]%self.period)/self.period)
        
        # Uniform cadence of the same size
        uniform = (np.arange(phased.size)+1)/phased.size

        # Differences between the CDFs of a real phased cadence and perfect uniform cadence
        d_plus = np.max(uniform - phased)
        d_minus = np.max(phased - uniform)
        
        # Kuiper test value is the sum of these two differences
        result = d_plus + d_minus

        return result