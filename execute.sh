#!/bin/bash

# Name of my job:
#PBS -N Monte_Carlo_Lstrings

# Run for 1 hour:
#PBS -l walltime=2:00:00

# Where to write stderr:
#PBS -e myprog.err

# Where to write stdout:
#PBS -o myprog.out

# Send me email when my job aborts, begins, or ends
#PBS -m abe

# Select number of processors
#PBS -l select=1:ncpus=16

# Load correct python
module load app/GEOS/3.6.2-gcc-4.4.7
module load python/3.5.0

# This command switched to the directory from which the "qsub" command was run:
cd $PBS_O_WORKDIR

# Set OMP_NUM_THREADS
export OMP_NUM_THREADS=16

#  Now run my program
python3 montepython_1.2.py

echo Done!