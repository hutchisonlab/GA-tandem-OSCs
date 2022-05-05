#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=0-05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=blp62@pitt.edu

source /ihome/crc/install/lmod/lmod/init/bash

# Load modules
module purge
module load openmpi/3.1.4
module load orca/4.2.0

inp=${SLURM_JOB_NAME}

ulimit -s unlimited

# Copy inputs to scratch
cp $SLURM_JOB_NAME $SLURM_SCRATCH
cd $SLURM_SCRATCH


# run the job, $(which orca) is necessary
$(which orca) $SLURM_JOB_NAME > /ihome/ghutchison/blp62/OPEP2/Calculations/donors/wb97x-d3_sTDA_cation/${SLURM_JOB_NAME%%.inp}.out

ls $SLURM_SCRATCH

# Copy outputs to submit directory
cp $SLURM_SCRATCH/*.prop /ihome/ghutchison/blp62/OPEP2/Calculations/donors/wb97x-d3_sTDA_cation/${SLURM_JOB_NAME%%.inp}.prop


