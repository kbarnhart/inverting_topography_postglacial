#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:45:14 2017

@author: barnhark
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os

os.system('dakota_restart_util to_tabular model_000_v2/lowering_history_0.pg24f_ic5etch/dakota_queso.rst model_000_v2/lowering_history_0.pg24f_ic5etch/dakota_queso_tabular.txt')

file = '/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM/model_000_v2/lowering_history_0.pg24f_ic5etch/dakota_queso_tabular.txt'
df = pd.read_csv(file, sep='\s+')

metric_col = [col for col in df if col.startswith('chi_elev')]

df['objective_function'] = np.sum(df.loc[:,metric_col]**2, axis=1)

plt.figure()
plt.title('Initial 40 samples for QUESO-DRAM: Model 000')
plt.scatter(df.K_sp_exp[:40], df.linear_diffusivity_exp[:40], c=df.objective_function[:40], vmin=df.objective_function.min(), vmax=1000 )
plt.xlim(-6, -1)
plt.xlabel('K_sp_exp')
plt.ylim(-6.3, -1.3)
plt.ylabel('linear_diffusivity_exp')
cb = plt.colorbar( )
cb.set_label('objective_function')
plt.savefig('initial_samples.png')

plt.show()


plt.figure()
plt.scatter(df.K_sp_exp[40:], df.linear_diffusivity_exp[40:], c=df.objective_function[40:])
plt.title('Refinement samples for QUESO-DRAM: Model 000')
plt.xlim(-6, -1)
plt.xlabel('K_sp_exp')
plt.ylim(-6.3, -1.3)
plt.ylabel('linear_diffusivity_exp')
cb = plt.colorbar( )
cb.set_label('objective_function')
plt.savefig('refinement_samples.png')
plt.show()

plt.figure()
plt.scatter(df.K_sp_exp, df.linear_diffusivity_exp, c=df.objective_function, vmax=2*(df.objective_function.min()))
plt.title('All samples for QUESO-DRAM: Model 000')
plt.xlim(-6, -1)
plt.xlabel('K_sp_exp')
plt.ylim(-6.3, -1.3)
plt.ylabel('linear_diffusivity_exp')
cb = plt.colorbar( )
cb.set_label('objective_function')
plt.savefig('all_samples.png')
plt.show()

#%%
plt.figure()
plt.scatter(df.K_sp_exp, df.linear_diffusivity_exp, c=df['%eval_id'])
plt.title('All samples for QUESO-DRAM: Model 000')
plt.xlim(-6, -1)
plt.xlabel('K_sp_exp')
plt.ylim(-6.3, -1.3)
plt.ylabel('linear_diffusivity_exp')
cb = plt.colorbar( )
cb.set_label('eval_id')
plt.savefig('all_samples_by_eval_id.png')
plt.show()
#%%
for value in range(50):
    key = '{0:02d}'.format(value)
    file = '/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM/model_000_v2_short/lowering_history_0.pg24f_ic5etch/'+key+'/posterior' + key + '.dat'
    if os.path.exists(file):
        posterior_df = pd.read_table(file, header=0, delimiter='\s+')
        
        metric_col = [col for col in posterior_df if col.startswith('chi_elev')]
        
        posterior_df['objective_function'] = np.sum(posterior_df.loc[:,metric_col]**2, axis=1)
        
        plt.figure()
        plt.scatter(posterior_df.K_sp_exp, posterior_df.linear_diffusivity_exp, c='k', s=1, alpha=0.003)
        plt.title('Posterior Points: '+key+' MCMC iterations')
        plt.xlim(-6, -1)
        plt.xlabel('K_sp_exp')
        plt.ylim(-6.3, -1.3)
        plt.ylabel('linear_diffusivity_exp')
        plt.savefig('posterior.'+key+'.iterations.png')
        plt.show()

        plt.figure()
        H, xedges, yedges = np.histogram2d(posterior_df.K_sp_exp, posterior_df.linear_diffusivity_exp, bins=70, normed=False, range=[[-6, -1], [-6.3, -1.3]])
        H_norm = H.T/posterior_df.linear_diffusivity_exp.size
        im = plt.imshow(H_norm, interpolation='nearest', origin='low',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap='magma_r', vmin=0, vmax=0.5)
        plt.title('Probability Density: '+key+' MCMC iterations')
        plt.xlabel('K_sp_exp')
        plt.ylabel('linear_diffusivity_exp')
        cb = plt.colorbar()
        cb.set_label('Probability')
        plt.savefig('density.'+key+'.iterations.png')
        plt.show()