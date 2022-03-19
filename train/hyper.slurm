#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Longhorn v100 nodes
#
#   *** Single Serial Job in v100 Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Longhorn login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#----------------------------------------------------
#SBATCH -J myjob           # Job name
#SBATCH -o hyper_search.o%j       # Name of stdout output file
#SBATCH -e GER.e%j       # Name of stderr error file
#SBATCH -p development            # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 112               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=ygqin@utexas.edu
#SBATCH -A ASC21002       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...
#module unload python3
conda activate torch_env
#module load python3
#export MY_SPECTRUM_OPTIONS="--gpu --aff on" 
module load launcher
module list
pwd
date

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=1node_train #para_test
#export LAUNCHER_JOB_FILE=para_test
#python3 disc_angle.py train 0
#python3 grainNN.py ini 0
#python3 grainNN.py train 0
#python3 grainNN.py test 0
$LAUNCHER_DIR/paramrun
