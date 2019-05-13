#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:01:39 2017

This script creates lowering histories for buttermilk creek based 
the current draft of the incision history report.  

@author: katybarnhart
"""
import csv
import numpy as np
import matplotlib.pylab as plt
plt.close('all')

header_text = ['time_yr', 'elev_change_since_model_start_ft']

# Option 1 Buttermilk Context

buttermilk_context_dates = np.asarray(    [-13000., -10600., -9495., -5632., -3785., -2300.,  -0.])
buttermilk_context_elevations = np.asarray([ 1340.,   1295.,  1285.,  1285.,  1223.,  1195.,  1181.])

buttermilk_context_time_yr = buttermilk_context_dates - buttermilk_context_dates[0]
buttermilk_context_elevation_change = buttermilk_context_elevations - buttermilk_context_elevations[0]

with open('lowering_history_1.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(buttermilk_context_time_yr.size):
        writer.writerow([buttermilk_context_time_yr[i], buttermilk_context_elevation_change[i]])


# Option 2 Meander 
meander_dates = np.asarray(    [-13000., -10600., -9495., -6764., -5632., -2500.,  -0.])
meander_elevations = np.asarray([ 1351.,   1301.,  1291.,  1291.,  1291.,  1192.,  1192.])

meander_time_yr = meander_dates - meander_dates[0]
meander_elevation_change = meander_elevations - meander_elevations[0]


with open('lowering_history_2.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(meander_time_yr.size):
        writer.writerow([meander_time_yr[i], meander_elevation_change[i]])


# Option 0 Average - for Calibration 

meander_sel = [0, 1, 2, 4, 5, 6]
bmilk_sel = [0, 1, 2, 3, 5, 6]
calib_time_yr = list((buttermilk_context_time_yr[bmilk_sel] + meander_time_yr[meander_sel])/2.)
calib_elevation_change = list((buttermilk_context_elevation_change[bmilk_sel] + meander_elevation_change[meander_sel])/2.)

# insert data point at -3785
calib_time_yr.insert(4,9215.000)
calib_elevation_change.insert(4, -117.000)
calib_dates = calib_time_yr + meander_dates[0]


with open('lowering_history_0.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_text)
    for i in range(len(calib_time_yr)):
        writer.writerow([calib_time_yr[i], calib_elevation_change[i]])

l1, = plt.plot(meander_dates, meander_elevations - meander_elevations[-1], 'goldenrod')
l2, = plt.plot(buttermilk_context_dates, buttermilk_context_elevations - buttermilk_context_elevations[-1], 'orangered')
l3, = plt.plot(calib_dates, calib_elevation_change -calib_elevation_change[-1], 'darkred')


#plt.plot(dates_inception_simple, elevations_simple, '-')

d1, = plt.plot(meander_dates, meander_elevations-meander_elevations[-1], 'k.')
d2, = plt.plot(buttermilk_context_dates, buttermilk_context_elevations-buttermilk_context_elevations[-1], 'b*')
ax = plt.gca()

#for i,j in zip(dates, elevations):
#    print(i)
#    offs = 0
#    if i == -9495.0:
#        offs = -20
#    text = '('+str(np.abs(i)/1000.)+' ka,\n '+str(j)+' ft)' 
#    ax.annotate(text,xy=(i,j+offs), size=8)
#i = inception_dates[0]
#j = elevations[0]
#text = '('+str(np.abs(i)/1000.)+' ka\n '+str(j)+' ft)' 
#ax.annotate(text,xy=(i-500,j-20), size=8)

leg = plt.legend([l1, l2, l3], ['Meander Scenario', 'Buttermilk Context Scenario', 'Average Scenario'])
ax = plt.gca().add_artist(leg)

plt.legend([d1, d2], ['Data Pairs: Meander Scenario', 'Data Pairs: Buttermilk Context Scenario'], loc=3)

#plt.title("Alternative Lowering Histories \n")
plt.xlabel("Years Before Present, Relative to 1950")
plt.ylabel("Elevation relative to modern channel (feet)")
plt.savefig('alternative_incision_histories.pdf')


plt.show()
