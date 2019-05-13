#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:33:29 2017

@author: barnhark
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

folder = '/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM/model_000/lowering_history_0.pg24f_ic5etch/'
fp = 'wv_model_000_surrogate_samples_01.dat'
df = pd.read_csv(folder+fp, sep='\s+')
size = int(df.shape[0]**0.5)

outputs = [col for col in df.columns if col.startswith('chi')]
df['objective_fxn'] = np.square(df[outputs]).sum(axis=1)


# actual points
fp = 'dakota_mcmc.dat'
df_complex = pd.read_csv(folder+fp, sep='\s+')

OF = df['objective_fxn'].reshape((size, size))
Y = df['linear_diffusivity_exp'].reshape((size, size))
X = df['K_sp_exp'].reshape((size, size))

plt.figure()
plt.pcolormesh(X,Y,np.log10(OF), cmap='magma_r')
plt.plot(df_complex['K_sp_exp'], df_complex['linear_diffusivity_exp'], 'c.')
plt.colorbar()
plt.xlabel('Ksp')
plt.ylabel('D')
plt.title('Model 000: \n Log10 of Objective Function Surrogate ')
