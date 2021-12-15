import gatspy.periodic
import numpy as np
import pandas as pd
import scipy.stats
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
        self.unique_filts_  = (parameters['unique_filts_'])
        self.ymean_by_filt_ = (parameters['ymean_by_filt_'])
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
