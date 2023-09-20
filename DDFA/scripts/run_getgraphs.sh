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

# Start singularity instance
python -u sastvd/scripts/getgraphs.py bigvul --sess --job_array_number $SLURM_ARRAY_TASK_ID --num_jobs 100 --overwrite
