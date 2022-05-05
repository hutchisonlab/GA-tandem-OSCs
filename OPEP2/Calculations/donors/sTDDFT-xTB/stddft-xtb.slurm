#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-5:00:00

# Load the module
module purge

export XTB4STDAHOME=/ihome/ghutchison/blp62/xtb4stda
export PATH=$PATH:${XTB4STDAHOME}/bin

# Copy files to SLURM_SCRATCH
cd $SLURM_SUBMIT_DIR
files=(
    *.xyz 
)
echo $files
for i in ${files[@]}; do
    echo "copying $i"
    cp $SLURM_SUBMIT_DIR/$i $SLURM_SCRATCH/$i
done

# cd to the SCRATCH space
cd $SLURM_SCRATCH

# run the job, $(which orca) is necessary
for inp in *.xyz; do
  base=${inp%%.xyz}
  echo "$inp"
  ${XTB4STDAHOME}/bin/xtb4stda ${inp} >${base}.out
  ${XTB4STDAHOME}/bin/stda -xtb -e 5 -rpa >${base}.stda 

  # copy incremental progress
  cp ${base}.out $SLURM_SUBMIT_DIR
  cp ${base}.stda $SLURM_SUBMIT_DIR
  rm -f *restart 
done
