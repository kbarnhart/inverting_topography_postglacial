# -*- coding: utf-8 -*-
"""
Driver model for Landlab Model 800 BasicRt

Katy Barnhart March 2017
"""
# import resource calculation modules and start logging usage
import resource, time
start_time = time.time()
usage_file = open('usage.txt', 'w')
usage_file.write(time.ctime()+'\n')

# import remaining required modules. 
import sys
import os
import shutil
from subprocess import call
from yaml import load

from erosion_model import BasicRt as Model
from metric_calculator import MetricDifference
from landlab import imshow_grid

import numpy as np

# set files and directories used to set input templates. 
# Files and directories.
start_dir = sys.argv[1]
input_file = 'inputs.txt'
input_template = 'inputs_template.txt'

# Use `dprepro` (from $DAKOTA_DIR/bin) to substitute parameter
# values from Dakota into the SWASH input template, creating a new
# inputs.txt file.
shutil.copy(os.path.join(start_dir, input_template), os.curdir)
call(['dprepro', sys.argv[2], input_template, input_file])
call(['rm', input_template])

# now prepare to run landlab. 
# load the params file to get the correct file names
with open(input_file, 'r+') as f:
    # load params file
    params = load(f)

random_seed = int(params['random_seed'])

# get filenames/etc. 
modern_dem_name = params['modern_dem_name']
outlet_id = params['outlet_id']
modern_dem_metric_file = params['modern_dem_metric_file']
modern_dem_chi_file = params['modern_dem_chi_file']
chi_mask_dem_name = params['chi_mask_dem_name']
outlet_id = params['outlet_id']

#plan for output files
output_fields =['topographic__elevation']

# write usage
usage = resource.getrusage(resource.RUSAGE_SELF)
usage_file.write('\n\nUsage Before Running Model: \n')
for name, desc in [
    ('ru_utime', 'User time'),
    ('ru_stime', 'System time'),
    ('ru_maxrss', 'Max. Resident Set Size'),
    ('ru_ixrss', 'Shared Memory Size'),
    ('ru_idrss', 'Unshared Memory Size'),
    ('ru_isrss', 'Stack Size'),
    ('ru_inblock', 'Block inputs'),
    ('ru_oublock', 'Block outputs'),
    ]:
    usage_file.write('%-25s (%-10s) = %s \n'%(desc, name, getattr(usage, name)))

#run the model
model = Model(input_file)

# modify the input topography based on random seed. 
nodes = model.grid.core_nodes
np.random.seed(random_seed)
topo_noise = np.random.normal(0, 0.3, (len(nodes)))
model.z[nodes] = model.z[nodes] + topo_noise

model.run(output_fields=output_fields)

usage = resource.getrusage(resource.RUSAGE_SELF)
usage_file.write('\n\nUsage After Running Model: \n')
for name, desc in [
    ('ru_utime', 'User time'),
    ('ru_stime', 'System time'),
    ('ru_maxrss', 'Max. Resident Set Size'),
    ('ru_ixrss', 'Shared Memory Size'),
    ('ru_idrss', 'Unshared Memory Size'),
    ('ru_isrss', 'Stack Size'),
    ('ru_inblock', 'Block inputs'),
    ('ru_oublock', 'Block outputs'),
    ]:
    usage_file.write('%-25s (%-10s) = %s \n'%(desc, name, getattr(usage, name)))

model_dem_name = model.params['output_filename'] + \
    str(model.iteration-1).zfill(4) + \
        '.nc'

# calculate metrics
md = MetricDifference(model_dem_name=model_dem_name,
                      modern_dem_metric_file = modern_dem_metric_file, 
                      modern_dem_chi_file = modern_dem_chi_file, 
                      outlet_id = outlet_id,
                      chi_mask_dem_name=chi_mask_dem_name)
md.run()

# write out metrics as "ouputs_for_analysis.txt' and as Dakota expects. 
output_bundle = md.dakota_bundle()
with open('outputs_for_analysis.txt', 'a') as fp:
    for metric in output_bundle:
        fp.write(str(metric)+'\n')

# write out residual. 
with open(sys.argv[3], 'w') as fp:
    for metric in output_bundle:
        fp.write(str(metric)+'\n')

cur_working = os.getcwd()
cur_working_split = cur_working.split(os.path.sep)
cur_working_split.append('png')
try:
    cut_ind = cur_working_split.index('results')+3
except:
    cut_ind = cur_working_split.index('study3py')+3

fig_name = '.'.join(cur_working_split[cut_ind:])

imshow_grid(model.grid, model.z, vmin=1230, vmax=1940, cmap='viridis', output=fig_name)

usage = resource.getrusage(resource.RUSAGE_SELF)
usage_file.write('\n\nUsage At End of Job: \n')
for name, desc in [
    ('ru_utime', 'User time'),
    ('ru_stime', 'System time'),
    ('ru_maxrss', 'Max. Resident Set Size'),
    ('ru_ixrss', 'Shared Memory Size'),
    ('ru_idrss', 'Unshared Memory Size'),
    ('ru_isrss', 'Stack Size'),
    ('ru_inblock', 'Block inputs'),
    ('ru_oublock', 'Block outputs'),
    ]:
    usage_file.write('%-25s (%-10s) = %s \n'%(desc, name, getattr(usage, name)))

end_time = time.time()
usage_file.write('\n\n'+time.ctime()+'\n')
usage_file.write('Elapsed Time: '+str(end_time-start_time)+'\n')

usage_file.close()
