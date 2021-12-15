import rubin_sim.maf as maf
import rubin_sim.utils as rsUtils
import numpy as np

class KuiperVS(maf.BaseMetric):
    def __init__(self, mjdCol='observationStartMJD', units='', period=5, **kwargs):
        """surveyLength = time span of survey (years) """
        """period = assumed period of variability which will be used for calculating phase coverage by\
        observations. Measured in days"""
        self.mjdCol = mjdCol
        super(KuiperVS, self).__init__(col=self.mjdCol, units=units, **kwargs)
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