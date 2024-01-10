# Create virtual environment
conda create --name deepdfa python=3.10 -y
conda activate deepdfa
# Install requirements
conda install cudatoolkit=11.8 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
pip install -r requirements.txt
# Add project files to import path
export PYTHONPATH="$PWD/DDFA:$PYTHONPATH"
