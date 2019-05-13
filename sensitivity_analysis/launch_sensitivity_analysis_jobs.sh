#!/bin/sh

#  Launch sensitivity analysis jobs.
#
#  Warning: This will launch XX jobs in order to make xx model evaluations.
#  This took XX core-hours on CU's Summit HPCC platform.
#
#  Created by Katherine Barnhart on 5/9/17.

# loop throug MOAT, DELSA; sew, gully
# so env is always set correctly

start_dir=$(pwd)

for area in 'sew' ; do

# construct all input templates and drivers.
cd $start_dir/$area
python make_input_templates_and_drivers.py

for method in 'MOAT' 'DELSA'; do

# cd to the correct place
cd $start_dir/$area/$method

# run the job creation script
sbatch launch_job_creation.sh

done
done
