#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:04:17 2017

@author: barnhark
"""

import pandas as pd
import os
import matplotlib.pylab as plt

fpl = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'climate_futures', 'maca_future_parameters.csv']

fp = os.path.join(os.path.sep, *fpl)

df = pd.read_csv(fp, index_col=[0,1], header=[0,1,2])


# climate future 1
with open('climate_future_1.constant_climate.txt', 'w') as f:
    f.write('opt_var_precip: False\n')

infiltration_capacity = df.loc[('mean', 'infiltration_capacity')][('value', 'Historical', '1970-1999')]

plt.figure()
# climate future 2
with open('climate_future_2.RCP45.txt', 'w') as f:
    f.write('opt_var_precip: True\n')
    f.write('precip_stop_time: 100.0\n')

    # write out historical parameters:
    for param in ['mean_storm__intensity', 'infiltration_capacity', 'precip_shape_factor', 'intermittency_factor']:
        pval = df.loc[('mean', param)][('value', 'Historical', '1970-1999')]
        f.write(param + ': ' + str(pval) + '\n')

    
    #intermittency_factor_rate_of_change is zero
    f.write('intermittency_factor_rate_of_change: ' + str(0.0) + '\n')

    #mean_depth_rate_of_change
    # 2050 value for RCP 4.5
    mean_depth_2000 = df.loc[('mean', 'mean_storm__intensity')][('value', 'Historical', '1970-1999')]
    mean_depth_2100 = df.loc[('mean', 'mean_storm__intensity')][('value', 'RCP 4.5', '2070-2099')]

    mean_depth_rate_of_change = (mean_depth_2050 - mean_depth_2000) / 100.0
    f.write('mean_depth_rate_of_change: ' + str(mean_depth_rate_of_change) + '\n')

plt.plot([0, 200], [mean_depth_2000, mean_depth_2000], 'g')
plt.plot([0, 100, 200], [mean_depth_2000, mean_depth_2100, mean_depth_2100], 'r')

# climate future 3
with open('climate_future_3.RCP85.txt', 'w') as f:
    f.write('opt_var_precip: True\n')
    f.write('precip_stop_time: 100.0\n')

    # write out historical parameters:
    for param in ['mean_storm__intensity', 'infiltration_capacity', 'precip_shape_factor', 'intermittency_factor']:
        pval = df.loc[('mean', param)][('value', 'Historical', '1970-1999')]
        f.write(param + ': ' + str(pval) + '\n')

    #intermittency_factor_rate_of_change is zero
    f.write('intermittency_factor_rate_of_change: ' + str(0.0) + '\n')

    #mean_depth_rate_of_change
    # 2100 value for RCP 8.5
    mean_depth_2000 = df.loc[('mean', 'mean_storm__intensity')][('value', 'Historical', '1970-1999')]
    mean_depth_2100 = df.loc[('mean', 'mean_storm__intensity')][('value', 'RCP 8.5', '2070-2099')]

    mean_depth_rate_of_change = (mean_depth_2100 - mean_depth_2000) / 100.0
    f.write('mean_depth_rate_of_change: ' + str(mean_depth_rate_of_change) + '\n')

plt.plot([0, 100, 200], [mean_depth_2000, mean_depth_2100, mean_depth_2100], 'b')

plt.legend(['No Change' ,'RCP 4.5', 'RCP 8.5'], title='Climate Scenario')
plt.xlabel('Time Since Future Run Onset')
plt.ylabel('mean_storm__intensity [m/yr]')
plt.savefig('ClimateFuture_mean_storm__intensity.png')

#mean_storm__intensity
#intermittency_factor
#precip_shape_factor
