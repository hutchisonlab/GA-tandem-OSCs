#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=0-05:00:00

# allows cron to run bash
#source /ihome/crc/install/lmod/lmod/init/bash

# Load modules
module purge

export XTBHOME=/ihome/ghutchison/oda6/xtb/xtb-641/bin/

inp=${SLURM_JOB_NAME}

# Copy files to SLURM_SCRATCH
cp $SLURM_JOB_NAME $SLURM_SCRATCH

# cd to the SCRATCH space
cd $SLURM_SCRATCH

base=${inp%%.xyz}
echo ${base}
# just getting the NFA file name
molname=${base##*/}
echo ${molname}

# run GFN2-xTB geometry optimization
${XTBHOME}/xtb $SLURM_JOB_NAME -opt > /ihome/ghutchison/blp62/OPEP2/Calculations/acceptors/GFN2_output/${molname}.out

# copy incremental progress
cp xtbopt.xyz /ihome/ghutchison/blp62/OPEP2/Calculations/acceptors/GFN2_optimized_xyz_for_benchmarking/3/${molname}.xyz
rm -f *restart 

# Load modules
module purge
module load intel openbabel

# create orca input file for sTD-DFT
obabel xtbopt.xyz -O .inp -o orcainp -m -xf /ihome/ghutchison/blp62/OPEP2/Calculations/wB97X-D3_def2SVP_sTDDFT_pp.txt

module purge
module load openmpi/3.1.4
module load orca/4.2.0


# run sTDDFT, $(which orca) is necessary
$(which orca) xtbopt.inp > /ihome/ghutchison/blp62/OPEP2/Calculations/acceptors/wb97x-d3_sTDDFT/${molname}.out

# Copy outputs to submit directory
cp $SLURM_SCRATCH/xtbopt.prop /ihome/ghutchison/blp62/OPEP2/Calculations/acceptors/wb97x-d3_sTDDFT/${molname}.prop






