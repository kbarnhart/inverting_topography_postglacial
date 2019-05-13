
#! /usr/bin/env python
# Brokers communication between Dakota and Landlab.
#
# Arguments:
#   $1 is 'params.in' from Dakota
#   $2 is 'results.out' returned to Dakota

"""Broker communication between Dakota and Landlab through files."""

import sys
import os
import re
import shutil
from subprocess import call

def write(results_file, array, labels):
    """Write a Dakota results file from an input array."""
    try:
        fp = open(results_file, 'w')
        for i in range(len(array)):
            fp.write(str(array[i]) + '\t' + labels[i] + '\n')
    except IOError:
        raise
    finally:
        fp.close()

def get_labels(params_file):
    """Extract labels from a Dakota parameters file."""
    labels = []
    try:
        fp = open(params_file, 'r')
        for line in fp:
            if re.search('ASV_', line):
                labels.append(''.join(re.findall(':(\S+)', line)))
    except IOError:
        raise
    finally:
        fp.close()
        return(labels)

# Files and directories.
start_dir = sys.argv[1]
labels = get_labels(sys.argv[2])
try:
    output_file = open('outputs_for_analysis.txt')
    #output_var = 'Ufric_x_002800_000' # final time step
except IOError:
    m_output = ['Nan', 'Nan']
else:
    lines = [line.rstrip('\n') for line in output_file]

    # Calculate the mean and standard deviation of the 'Botlev' output
    # values for the simulation. Write the output to a Dakota results
    # file.
    #output = read(output_file, output_var)
    if lines: #equivalent to checking if list has content
        m_output = lines
    else:
        m_output = [-99, -99] #-99 represents a 'bad' value.
write(sys.argv[3], m_output, labels)
