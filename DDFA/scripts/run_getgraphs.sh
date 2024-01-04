#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-99%10
#SBATCH --err="logs/getgraphs_%A_%a.out"
#SBATCH --output="logs/getgraphs_%A_%a.out"
#SBATCH --job-name="getgraphs"

source activate.sh

if [ ! -z "$SLURM_ARRAY_TASK_ID"]
then
    jan="--job_array_number $SLURM_ARRAY_TASK_ID"
else
    jan=""
fi

# Start singularity instance
python -u sastvd/scripts/getgraphs.py bigvul --sess $jan --num_jobs 100 --overwrite $@
