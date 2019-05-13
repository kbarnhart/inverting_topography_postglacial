#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:07:26 2018

@author: barnhark
"""
import os
import glob
import yaml
import numpy as np
import matplotlib.pylab as plt

from erosion_model import PrecipChanger

climate_future_folderpath = ['..', '..', '..', 'auxillary_inputs', 'climate_futures']

climate_futures = glob.glob(os.path.abspath(os.path.join(*(climate_future_folderpath+['climate_future*.txt']))))

DAYS_PER_YEAR = 365.25
length_factor = 3.28084
#%%
for climate_future in climate_futures:

    with open(climate_future, 'r') as f:
        climate_vars = yaml.load(f)
        
        
    if climate_vars['opt_var_precip']:    
        frac_wet_days = climate_vars['intermittency_factor']
        frac_wet_rate = climate_vars['intermittency_factor_rate_of_change']
    
        # get mean storm intensity and rate of change from parameters
        # these have units of length per time, so convert using the length
        # factor
        mdd = climate_vars['mean_storm__intensity'] / DAYS_PER_YEAR * length_factor
        mdd_roc = climate_vars['mean_depth_rate_of_change'] / DAYS_PER_YEAR * length_factor
    
        # get precip shape factor.
        c = climate_vars['precip_shape_factor']
        #c = 0.77

        # if infiltration capacity is provided, set it.
        #

        try:
            ic = climate_vars['infiltration_capacity'] * length_factor
           # ic = 5.5*length_factor
        except KeyError:
            ic = None
    
        # if m_sp is provided, set it
        try:
            m = climate_vars['m_sp']
        except KeyError:
            m = None
    
        # if precip-stop time is provided, set it, otherwise use the
        # total run time.
        try:
            stop_time = climate_vars['precip_stop_time']
    
        except KeyError:
            stop_time = climate_vars['run_duration']    
        
        m = 0.5
        
        
        
        pc = PrecipChanger(starting_frac_wet_days=frac_wet_days,
                           frac_wet_days_rate_of_change=frac_wet_rate,
                           starting_daily_mean_depth=mdd,
                           mean_depth_rate_of_change=mdd_roc,
                           precip_shape_factor=c,
                           time_unit='year',
                           infiltration_capacity=ic,
                           m=m,
                           stop_time=stop_time)
    
        
        time = np.arange(0, 10001, 10)
        adjustment_factor = [pc.get_erodibility_adjustment_factor(t) for t in time]
        
        
        erodability_adjustment_rate = (adjustment_factor[2] - adjustment_factor [1]) / (time[2] - time[1])
        print(erodability_adjustment_rate)    
        print(os.path.split(climate_future)[-1], adjustment_factor[-1])
        plt.figure()
        plt.plot(time, adjustment_factor, '.-')
        plt.xlabel('time')
        plt.ylabel('erodability adjustment factor')
        plt.title(os.path.split(climate_future)[-1])
        plt.xlim(0, 250)
        plt.ylim(0.99, 1.25)
        plt.savefig(os.path.split(climate_future)[-1]+'.should_have_used.png')