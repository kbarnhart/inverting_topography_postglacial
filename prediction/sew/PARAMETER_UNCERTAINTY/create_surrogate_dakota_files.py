#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 08:05:17 2018

@author: barnhark
"""

import pandas as pd
import os
import glob
import numpy as np

models = glob.glob('model_*')

# get the names of the output locations
plot_locations = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew', 'PredictionPoints_ShortList.csv'])
plot_location_df = pd.read_csv(plot_locations)

timesteps = np.arange(0, 101)

metric_names = []
loc_names = np.sort(plot_location_df.Point_Name)
for loc_name in loc_names:
    mets = [loc_name+'.'+str(time) for time in timesteps]
    metric_names.extend(mets)

# Read in the Dakota template file
with open('dakota_create_and_sample_surrogate_template.in', 'r') as f:
    dakota_template_lines = f.readlines()

dakota_cmnd = 'dakota -i dakota_create_and_sample_surrogate.in -o dakota_create_and_sample_surrogate.out --write_restart dakota_surrogate_pred.rst &> dakota_surrogate.log'

#%%

sbatch_commands = []

# for model in model
for model in models:

    # get posterior file, and write out a simplified version

    # set filenames
    posterior_in = os.path.join(os.path.sep, *['work/', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', 'sew', 'QUESO_DRAM', model, 'lowering_history_0.pg24f_ic5etch', 'posterior.dat'])
    posterior_out = os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew', 'PARAMETER_UNCERTAINTY', model, 'posterior.dat'])


    # read in data file
    posterior_dat = pd.read_csv(posterior_in, sep='\s+')

    # ID columns to keep
    keep = [col for col in posterior_dat.columns.values if not (col.startswith('chi_elev') or
                                                                col.startswith('interface') or
                                                                col.endswith('_id'))]

    # select right part of df
    posterior_dat_sel = posterior_dat[keep]

    # save df.
    posterior_dat_sel.to_csv(posterior_out, sep = '\t', header=True, index=False)


    # get posterior mins and maxes to make sure to set bounds on dakota file correctly.
    mins = posterior_dat_sel.min()
    maxs = posterior_dat_sel.max()


    # for lowering/climate
    for boundaries in glob.glob(os.path.join(model, '*') + os.path.sep):

        samples_file = os.path.join(boundaries, 'wv_' + model + '_prediction_sampling.dat')

        if not os.path.exists(samples_file):
            print('skipping          : '+ boundaries)

        if os.path.exists(samples_file):
            print('creating files for: '+ boundaries)
            # get all the complex samples
            samples = pd.read_csv(samples_file, sep = '\s+')

            keep_always = np.arange(2,np.where(samples.columns == 'ErdmanEdge.0')[0])

            cmnd_lines = []
            # for each location
            for location in loc_names:

                # ID columns to keep this time.
                responses_ids = [i for i in range(samples.columns.values.size) if samples.columns.values[i].startswith(location)]
                responses_names = [repr(r) for r in samples.columns.values[responses_ids]]
                keep_this_time = np.concatenate((keep_always, responses_ids))

                # create a directory for this work to occur in
                work_dir = os.path.join(boundaries, 'surrogates', location)
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)

                # subset the complex samples to just consider those in this location
                keep_df = samples.iloc[:,keep_this_time]

                # write those samples out to a file.
                keep_df.to_csv(os.path.join(work_dir, 'complex_samples.dat'), header=True, index=False, sep = '\t')


                upper_bounds = [str(np.ceil(m)) for m in maxs]
                lower_bounds = [str(np.floor(m)) for m in mins]

                variable_names = [repr(v) for v in mins.index.values]

                # now update the dakota template and write it to the work diretory.
                dakota_file = []
                for line in dakota_template_lines:
                    line = line.replace('{model_name}', model)
                    line = line.replace('{boundaries}', boundaries.split(os.path.sep)[1])
                    line = line.replace('{location}', location)
                    line = line.replace('{num_variables}', str(len(mins)))
                    line = line.replace('{variable_names}', ' '.join(variable_names))
                    line = line.replace('{lower_bounds}', ' '.join(lower_bounds))
                    line = line.replace('{upper_bounds}', ' '.join(upper_bounds))
                    line = line.replace('{response_names}', ' '.join(responses_names))

                    dakota_file.append(line)
                # Write dakota input file
                with open(os.path.join(work_dir, 'dakota_create_and_sample_surrogate.in'), 'w') as dakota_f:
                    dakota_f.writelines(dakota_file)

                cmnd_lines.append('cd '+ os.path.abspath(work_dir) + '; ' + dakota_cmnd + '\n')

            cmnd_lines_file = os.path.abspath(os.path.join(boundaries, 'cmnd_lines'))
            with open(cmnd_lines_file, 'w') as f:
                for line in cmnd_lines:
                    f.write(line)

            # create a sbatch file to evaluate these comandlines.
            surrogate_eval_file = os.path.abspath(os.path.join(boundaries,'submit_surrogate_evaluations.sh'))
            sbatch_commands.append(surrogate_eval_file)
            with open(surrogate_eval_file, 'w') as f:
                script = ['#!/bin/sh',
                          '#SBATCH --job-name surr_'+model,
                          '#SBATCH --ntasks-per-node 24',
                          '#SBATCH --partition shas',
                          '#SBATCH --mem-per-cpu 4GB',
                          '#SBATCH --nodes 1',
                          '#SBATCH --time 24:00:00',
                          '#SBATCH --account ucb19_summit1',
                          '',
                          '# load environment modules',
                          'module purge',
                          'module load intel/16.0.3',
                          'module load openmpi/1.10.2',
                          'module load cmake/3.5.2',
                          '#module load perl',
                          'module load mkl',
                          'module load gsl',
                          '## load_balance related modules',
                          '#module load impi',
                          '#module load loadbalance',
                          '',
                          '# make sure environment variables are set correctly',
                          'source ~/.bash_profile',
                          '',
                          'source ' + cmnd_lines_file,
                          '#mpirun lb '+ cmnd_lines_file]

                for line in script:
                    f.write(line+'\n')

with open('submit_all_surrogate_evals.sh', 'w') as f:
    for line in sbatch_commands:
        f.write('sbatch ' + line + '\n')
