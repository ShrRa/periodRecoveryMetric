import pandas as pd
import numpy as np
import gatspy.periodic
from .gatspy_extension import ExtendedLS
needed_row_columns=['lightcurve name','source_id','input period','Type','Subtype']
def fit_lightcurve_input_tester(inputs,adapted_dataset):
            
            i,Nterms,row= inputs
            D = pd.read_csv(row['lightcurve name'],index_col=0)
            
            model = gatspy.periodic.LombScargleMultiband(fit_period=True,
                                                             optimizer_kwds=dict(quiet=True),
                                                             Nterms_base=Nterms)
            model.optimizer.period_range=(D['Range min'],D['Range max'])
            
            model.fit(D.t,D.mag,D.magerr, D.filt)      
            return True
            #bestpers = model.find_best_periods(10,True)
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
            m.import_parameters(m.get_parameters(model))
            m.copy_parameters(model)
            params= m.export_parameters()
            params['source_id'] = row['source_id']
            params['Expected'] = row['input period']
            params['E-C'] = params['Expected']-params['_best_period']
            params['Type'] = row['Type']
            params['Subtype'] = row['Subtype']
            params['Nterms'] = Nterms
            #D = self.get_adapted_lightcurve(row)
            return params
def fit_lightcurve(inputs,adapted_dataset):
            
            i,Nterms,row= inputs
            D = pd.read_csv(row['lightcurve name'],index_col=0)
            model = gatspy.periodic.LombScargleMultiband(fit_period=True,
                                                             optimizer_kwds=dict(quiet=True),
                                                             Nterms_base=Nterms)
            model.optimizer.period_range=(D['Range min'],D['Range max'])
            model.fit(D.t,D.mag,D.magerr, D.filt)        
            #bestpers = model.find_best_periods(10,True)
            m = ExtendedLS(fit_period=True,optimizer_kwds=dict(quiet=True),Nterms_base=Nterms)
            m.import_parameters(m.get_parameters(model))
            m.copy_parameters(model)
            params= m.export_parameters()
            params['source_id'] = row['source_id']
            params['Expected'] = row['input period']
            params['E-C'] = params['Expected']-params['_best_period']
            params['Type'] = row['Type']
            params['Subtype'] = row['Subtype']
            params['Nterms'] = Nterms
            #D = self.get_adapted_lightcurve(row)
            return params