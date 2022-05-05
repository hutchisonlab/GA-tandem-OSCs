#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-05:00:00
#SBATCH --output=ga6_out.out
#SBATCH --mail-user=blp62@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge
module load python/ondemand-jupyter-python3.8

ulimit -s unlimited


source activate /ihome/ghutchison/blp62/.conda/envs/sklearn-env
python GA6_tandem_2.py