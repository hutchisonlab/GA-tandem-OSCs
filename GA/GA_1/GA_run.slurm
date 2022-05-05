#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6-00:00:00
#SBATCH --output=ga1_out.out
#SBATCH --mail-user=blp62@pitt.edu
#SBATCH --mail-type=END,FAIL

module purge
#module load python/anaconda3.5-4.2.0
module load python/ondemand-jupyter-python3.8

#source /ihome/ghutchison/blp62/.conda/envs/py35_opep/bin/activate

source activate /ihome/ghutchison/blp62/.conda/envs/sklearn-env
python GA_NFA_fused_core_3units.py