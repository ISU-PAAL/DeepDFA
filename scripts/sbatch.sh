#!/bin/bash
# Run a script with GPU on slurm

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 16G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --err="sbatch_%j.info"
#SBATCH --output="sbatch_%j.info"
#SBATCH --job-name="sbatch"


source activate.sh
module load gcc/10.2.0-zuvaafu cuda/11.3.1-z4twu5r
nvidia-smi

which python
python -V
which pip
pip -V

log_filename="slurmlog_$(echo $@ | sed -e 's@ @-@g' -e 's@/@-@g').log"
echo $log_filename

echo "command: $@"
$@ 2>&1 | tee $log_filename
