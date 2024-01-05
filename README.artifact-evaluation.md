# Purpose

This artifact includes the code to reproduce our experiments on DeepDFA, LineVul, and LineVul+DeepDFA, accepted at ICSE 2024.
This constitutes a large part of the research prototype, with other experiments consisting of running other models and running on other datasets.
Instructions for running other experiments are included in README.extended.md inside the repository.

We are applying for the following badges:
- **Available**: We placed the code in a publicly accessible archival repository in Github and on Figshare, at the URL https://doi.org/10.6084/m9.figshare.21225413.
- **Reusable**: We included code to run the full pre-processing end-to-end, starting from the source code in the raw Big-Vul dataset and ending with the performance results reported in the paper. We refactored the code and data pipelines and distributed the pre-processed data to encourage re-use and extension.

# Provenance

Download code from the GitHub repo: https://github.com/ISU-PAAL/DeepDFA.

Download the data from Figshare: https://doi.org/10.6084/m9.figshare.21225413.

Other links:
- Paper: "Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection" https://www.computer.org/csdl/proceedings-article/icse/2024/021700a166/1RLIWqviwEM
  - [Also see PDF here](./paper.pdf)
- ArXiv preprint: https://arxiv.org/abs/2212.08108

# Data

- Link to dataset: https://doi.org/10.6084/m9.figshare.21225413
- Link to Big-Vul which we used:
  - Repository: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
  - Raw dataset: https://drive.google.com/uc?id=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X
- Link to LineVul's version of Big-Vul dataset which we used to run LineVul for baseline and DeepDFA+LineVul experiments:
  - Repository: https://github.com/awsm-research/LineVul
  - test: https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
  - train: https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
  - val: https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ

# Setup

- Hardware: We ran the experiments on an AMD Ryzen 5 1600 3.2 GHz processor with 48GB of RAM and an Nvidia 3090 GPU with 24GB of GPU memory and CUDA 11.3.
- Software: Ubuntu Linux 22.04.

# Usage

Use these scripts to run the main performance experiments from the paper (Table 3b).
See README.extended.md in the Github repo for the extended instructions on running other experiments.

## Get the code

```bash
git clone https://github.com/ISU-PAAL/DeepDFA
cd DeepDFA
```

## Set up dependencies

```bash
# In repository root directory

# Create virtual environment
conda create --name deepdfa python=3.10 -y
conda activate deepdfa
# Install requirements
conda install cudatoolkit=11.6 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
pip install -r requirements.txt
# Add project files to import path
export PYTHONPATH="$PWD/DDFA:$PYTHONPATH"
# Install joern and add it to the executable path
bash scripts/install_joern.sh
export PATH="$PWD/joern/joern-cli:$PATH"
```

## Unpack data

```bash
# In repository root directory
bash scripts/download_all.sh
```

## Train DeepDFA

This script trains DeepDFA based on the dataset of source code already processed into CFGs.
It reports the performance at the end, comparable to Table 3b in our paper.

```bash
cd DDFA
# Train DeepDFA
bash scripts/train.sh --seed_everything 1
```

## Train LineVul baseline or DeepDFA+LineVul

These scripts report performance of the LineVul and DeepDFA+LineVul models, comparable to Table 3b in our paper.

```bash
cd LineVul/linevul
# Train LineVul
bash scripts/msr_train_linevul.sh 1 MSR
# Train DeepDFA+LineVul
bash scripts/msr_train_combined.sh 1 MSR
```

## Run end-to-end processing

The above scripts use the preprocessed data included in our data archive, for ease of replicability. The instructions below show how to run the code end-to-end.

### On sample data

The current prototype scripts take some time to process data into the format for our dataset, so we provide instructions how to do it with sample mode or full data mode.

```bash
cd DDFA
bash scripts/run_prepare.sh --sample
bash scripts/run_getgraphs.sh --sample
bash scripts/run_dbize.sh --sample
bash scripts/run_abstract_dataflow.sh --sample
bash scripts/run_absdf.sh --sample
# Train DeepDFA
bash scripts/train.sh --seed_everything 1 --data.sample_mode True

cd ../LineVul/linevul
# Train DeepDFA+LineVul
bash scripts/msr_train_combined.sh 1 MSR --sample --epochs 1
```

### Full data preprocessing

To run the full preprocessing from the raw MSR dataset to training DeepDFA, unpack only `storage/external/MSR_data_cleaned.csv` (skipping the preprocessed data such as the CFGs in `storage/processed`) and run these steps.

```bash
cd DDFA
bash scripts/run_prepare.sh
bash scripts/run_getgraphs.sh # Make sure Joern is installed!
bash scripts/run_dbize.sh
bash scripts/run_abstract_dataflow.sh
bash scripts/run_absdf.sh
# Train DeepDFA
bash scripts/train.sh --seed_everything 1
```
