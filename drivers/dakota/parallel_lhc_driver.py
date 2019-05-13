# -*- coding: utf-8 -*-
"""
Driver for running parallel dakota runs.

"""

import sys
import os
import shutil
from subprocess import call

# Files and directories.
start_dir = sys.argv[1]
print(start_dir)
input_file = 'dakota_centered_modeling.in'
output_file = 'dakota_centered_modeling.out'

dakota_template = 'dakota_centered_template.in'

# Use `dprepro` (from $DAKOTA_DIR/bin) to substitute parameter
# values from Dakota into the SWASH input template, creating a new
# inputs.txt file.
shutil.copy(os.path.join(start_dir, dakota_template), os.curdir)
call(['dprepro', sys.argv[2], dakota_template, input_file])

# call new dakota file.
with open('run.log', "w") as file_out:
    call(['dakota', '-i', input_file , '-o', output_file])

# Write temp out file so dakota will move on.
output_filepath = os.path.abspath(os.path.join(os.curdir, sys.argv[3]))
with open(output_filepath, 'w') as fp:
    nmetrics = 21
    for m in range(nmetrics):
        fp.write(str(m) + '\n')
