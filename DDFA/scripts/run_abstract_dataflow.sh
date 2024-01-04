#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/absdf.log
#SBATCH --error=logs/absdf.log
#SBATCH --job-name="absdf"

source activate.sh

set -e

python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --no-cache --stage 1 $@
python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --no-cache --stage 2 $@
