#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH --output=ga5_out_smear12.out
#SBATCH --mail-user=blp62@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge
module load python/ondemand-jupyter-python3.8

ulimit -s unlimited


source activate /ihome/ghutchison/blp62/.conda/envs/sklearn-env
python GA5_tandem_1_smear12.py