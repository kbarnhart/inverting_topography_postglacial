import sys
import os

os.system('squeue --user=barnhark > run.log')

with open('run.log', 'r') as f:
    lines = f.readlines()

print(len(lines))

