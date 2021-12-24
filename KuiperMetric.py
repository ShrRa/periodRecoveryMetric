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
    
    Returns
    -------
    Kuiper value in range [0;1] where 0 means perfectly uniform distribution and 1 means delta-function
    """
    
    def __init__(self, mjdCol='observationStartMJD', period=1., **kwargs):
        
       
        
        """period = assumed period of variability which will be used for calculating phase coverage by\
        observations. Measured in days"""
        self.mjdCol = mjdCol
        super(KuiperVS, self).__init__(col=self.mjdCol, units='Kuiper value, 0-1', **kwargs)
        self.period=period

    def run(self, dataSlice, slicePoint=None):
        # If only one observation, it's delta function and DKuiper=1
        if dataSlice[self.mjdCol].size == 1:
            return 1
        # Create phased cadence
        phased=np.sort((dataSlice[self.mjdCol]%self.period)/self.period)
        
        # Uniform cadence of the same size
        uniform = (np.arange(phased.size)+1)/phased.size

        # Differences between the CDFs of a real phased cadence and perfect uniform cadence
        d_plus = np.max(uniform - phased)
        d_minus = np.max(phased - uniform)
        
        # Kuiper test value is the sum of these two differences
        result = d_plus + d_minus

        return result