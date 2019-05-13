# -*- coding: utf-8 -*-
"""
Driver for running parallel dakota runs.

"""

import sys
import os
import shutil
from subprocess import call

#print('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))
#sys.stdout.flush()

# Files and directories.
start_dir = sys.argv[1]
input_file = 'inputs.txt'
input_template = 'inputs_template.txt'
run_script = 'driver.py'

# Use `dprepro` (from $DAKOTA_DIR/bin) to substitute parameter
# values from Dakota into the SWASH input template, creating a new
# inputs.txt file.
shutil.copy(os.path.join(start_dir, input_template), os.curdir)
call(['dprepro', sys.argv[2], input_template, input_file])
call(['rm', input_template])

# Write command to submit to Slurm into the file cmd_lines
# returns immediately, so jobs do not block.
# job_name = 'WVLL-Dakota' + os.path.splitext(os.getcwd())[-1]

cur_dir = os.path.abspath(os.getcwd())

full_path_to_run_script = os.path.abspath(os.path.join(start_dir, run_script))
job_command = 'cd '+ cur_dir + '; python ' + full_path_to_run_script + '\n'
with open(os.path.abspath(os.path.join(start_dir, 'cmd_lines_all')), "a") as myfile:
    myfile.write(job_command)

#call(['qsub', '-N', job_name, os.path.join(start_dir, run_script)])

# Write temp out file so dakota will move on.
with open(sys.argv[3], 'w') as fp:
    nmetrics = 32
    for m in range(nmetrics):
        fp.write(str(m) + '\n')
