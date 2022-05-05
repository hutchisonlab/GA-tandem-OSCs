#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=blp62@pitt.edu
#SBATCH --time=0-23:00:00

# Load the module
module purge

export XTBHOME=/ihome/ghutchison/oda6/xtb/xtb-641/bin/

inp=${SLURM_JOB_NAME}

# Copy files to SLURM_SCRATCH
cd $SLURM_SUBMIT_DIR
files=(
    *.mol 
)
echo $files
for i in ${files[@]}; do
    echo "copying $i"
    cp $SLURM_SUBMIT_DIR/$i $SLURM_SCRATCH/$i
done

# cd to the SCRATCH space
cd $SLURM_SCRATCH

# run the job, $(which orca) is necessary
for inp in *.mol ; do
  base=${inp%%.mol}
  ls -l ${inp}
  ${XTBHOME}/xtb ${inp} --sp --alpb hexane >${base}.out

  # copy incremental progress
  cp ${base}*.out $SLURM_SUBMIT_DIR
  rm -f *restart 
done
