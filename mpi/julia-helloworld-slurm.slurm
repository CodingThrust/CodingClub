#!/bin/bash

############## SLURM SETUP ###############
#SBATCH --job-name=mpi_job
#SBATCH --ntasks=128
# less than 10 hours
#SBATCH --time=10:00:00
#SBATCH --output=log.%j

############## MODULE LOADING ###############
# Clear the environment from any previously loaded modules
# module purge > /dev/null 2>&1
# Show available modules with module avail
module load openmpi

############## RUN ######################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun $HOME/.local/bin/julia --project=$PWD mpi.jl
