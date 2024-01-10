#!/bin/bash
set -e

conda create -f environment.yml
conda activate deepdfa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/DDFA:$PYTHONPATH"
