#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:01:39 2017

This script creates lowering futures for buttermilk creek

•	Scenario 1 (S1) assumes Buttermilk Creek will continue to incise at its 
current rate of 0.005 ft/yr. Using this assumption, the river will incise an 
additional 50 feet of the second resistant unit during the next 10,000 year 
period.

•	Scenario 2 (S2) assumes Buttermilk Creek will incise at a rate of 0.012 
feet per year, which is the average rate of incision over the past 13,000 year 
period. Using this assumption, the river will incise an additional 120 feet 
during the next 10,000 years.

•	Scenario 3 (S3) assumes Buttermilk Creek will incise at a rate of 0.025 
ft/yr, which is the average rate of incision through the less-resistant shale 
sections of the time-verses-elevation plot over the past 13,000 year period. 
In other words, this scenario assumes the second resistant unit will be 
composed of less resistant shale units for the next 10,000 years. Using this 
assumption, the river will incise an additional 250 feet during the next 
10,000 years. 


@author: katybarnhart
"""
import csv
import numpy as np
import matplotlib.pylab as plt
plt.close('all')

header_text = ['time_yr', 'elev_change_since_model_start_ft']

current_rate = 0.005
average_rate = 0.012

shale_rate = 0.025

# Scenario 1
lf1_time_yr = np.asarray([0, 10000.])
lf1_elevation_change = np.asarray([ 0.,   -10000. * current_rate])

with open('lowering_future_1.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(lf1_time_yr.size):
        writer.writerow([lf1_time_yr[i], lf1_elevation_change[i]])


# Scenario 2
lf2_time_yr = np.asarray([0, 10000.])
lf2_elevation_change = np.asarray([ 0.,  -10000. * average_rate ])

with open('lowering_future_2.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(lf2_time_yr.size):
        writer.writerow([lf2_time_yr[i], lf2_elevation_change[i]])

## Scenario 3
lf3_time_yr = np.asarray([0, 10000.])
lf3_elevation_change = np.asarray([ 0.,  -10000. * shale_rate])

with open('lowering_future_3.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(lf3_time_yr.size):
        writer.writerow([lf3_time_yr[i], lf3_elevation_change[i]])

#%% get past values for plotting comparison:
    
# Option 1 Buttermilk Context
buttermilk_context_dates = np.asarray([-13000., -10600., -9495., -5632., -3785., -2300.,  -0.])
buttermilk_context_elevations = np.asarray([ 1340.,   1295.,  1285.,  1285.,  1223.,  1195.,  1181.])

buttermilk_context_time_yr = buttermilk_context_dates - buttermilk_context_dates[0]
buttermilk_context_elevation_change = buttermilk_context_elevations - buttermilk_context_elevations[0]

# Option 2 Meander 
meander_dates = np.asarray([-13000., -10600., -9495., -6764., -5632., -2500.,  -0.])
meander_elevations = np.asarray([ 1351.,   1301.,  1291.,  1291.,  1291.,  1192.,  1192.])

meander_time_yr = meander_dates - meander_dates[0]
meander_elevation_change = meander_elevations - meander_elevations[0]

# Option 0 Average - for Calibration 
meander_sel = [0, 1, 2, 4, 5, 6]
bmilk_sel = [0, 1, 2, 3, 5, 6]
calib_time_yr = list((buttermilk_context_time_yr[bmilk_sel] + meander_time_yr[meander_sel])/2.)
calib_elevation_change = list((buttermilk_context_elevation_change[bmilk_sel] + meander_elevation_change[meander_sel])/2.)

# insert data point at -3785
calib_time_yr.insert(4,9215.000)
calib_elevation_change.insert(4, -117.000)
calib_dates = calib_time_yr + meander_dates[0]

plt.figure()
plt.plot([-13000, 10000], [0, 0], ':', color='dimgray', alpha=0.5)
#plt.plot([-13000, 10000], [-30, -30], ':', color='dimgray', alpha=0.5)
plt.plot([-13000, 10000], [-530, -530], '-', color='dimgray', alpha=0.5)

l1, = plt.plot(meander_dates, meander_elevations - meander_elevations[-1], 'goldenrod')
l2, = plt.plot(buttermilk_context_dates, buttermilk_context_elevations - buttermilk_context_elevations[-1], 'orangered')
l3, = plt.plot(calib_dates, calib_elevation_change -calib_elevation_change[-1], 'darkred')

l4, = plt.plot(lf1_time_yr, lf1_elevation_change, 'navy')
#l5, = plt.plot(lf2_time_yr, lf2_elevation_change, 'steelblue')
l6, = plt.plot(lf2_time_yr, lf2_elevation_change, 'darkolivegreen')
#l7, = plt.plot(lf4_time_yr, lf4_elevation_change, 'darkmagenta')
l8, = plt.plot(lf3_time_yr, lf3_elevation_change, 'yellowgreen')
#plt.plot(dates_inception_simple, elevations_simple, '-')

ax = plt.gca()

plt.text( 0.87, .07, 
         'Lake Erie Level', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

plt.text( 0.2, .77, 
         'Modern Elevation', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

leg = plt.legend([l1, l2, l3], ['Meander', 'Buttermilk Context', 'Average'], title="Past Scenarios", loc=3)
ax = plt.gca().add_artist(leg)

plt.legend([l4, l6, l8], ['S1', 'S2', 'S3'], loc=8, title="Future Scenarios", bbox_to_anchor=(0.55, 0))



#plt.title("Alternative Lowering Histories \n")
plt.xlabel("Years Before Present, Relative to 1950")
plt.ylabel("Elevation relative to modern channel (feet)")
plt.savefig('alternative_incision_histories_and_futures.pdf')

plt.show()