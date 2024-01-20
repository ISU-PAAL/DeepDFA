# Reproduction package for "Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection"

This artifact includes the code to reproduce our experiments on DeepDFA, LineVul, and LineVul+DeepDFA, accepted at ICSE 2024.
This constitutes a large part of the research prototype, with other experiments consisting of running other models and running on other datasets.

Links:
- Paper: "Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection" https://www.computer.org/csdl/proceedings-article/icse/2024/021700a166/1RLIWqviwEM
  - [Also see PDF here](./paper.pdf)
- ArXiv preprint: https://arxiv.org/abs/2212.08108
* Data package: https://doi.org/10.6084/m9.figshare.21225413
* GitHub repo: https://github.com/ISU-PAAL/DeepDFA

# Changelog

- Initial data package creation: September 20, 2023
- Cleanup for artifact evaluation: January 04, 2024
  - Add full usage instructions and scripts
  - Fix bugs
- Integrate feedback from artifact evaluation: January 10, 2024
  - Add documentation and convenience scripts
  - Add Docker container

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

## Schema

This section describes the organization and contents of various datasets and preprocessed files.
Datasets and preprocessed files must be downloaded from Figshare and unpacked into specific locations in the repository (using [scripts/download_all.sh](./scripts/download_all.sh) and the related [download instructions](#unpack-data)).
The total storage requirements are about 8GB download bandwidth and 45 GB unzipped storage space.

- Repository organization:
  - `CodeT5`: the code for the CodeT5 and CodeT5+DeepDFA models.
  - `DDFA`: the code for the DeepDFA model.
  - `LineVul`: the code for the LineVul and LineVul+DeepDFA models.
  - `scripts`: miscellaneous scripts we used to report the results.
  - `README.md`: this file
- Datasets in `DDFA/storage/external` and `LineVul/data/MSR`:
  - `DDFA/storage/external/MSR_data_cleaned.csv`: The original Big-Vul dataset. To unzip this file, unzip `MSR_data_cleaned.zip` in `DDFA/storage/external/`.
  - `DDFA/storage/external/linevul_splits.csv`: A manifest of which dataset partition (training, validation, or test) contains each example in the Big-Vul dataset according to LineVul's random split, which we reused for our experiments.
  - `LineVul/data/MSR/{train,val,test}.csv`: The Big-Vul dataset, preprocessed for consumption by LineVul. To unzip this file, unzip `MSR_LineVul.zip` in `LineVul/data/MSR/`.
- Cache files in `DDFA/storage/cache`:
  - `DDFA/storage/cache/bigvul_valid_False.csv`: The CFGs produced by Joern can have some irregularities which make them invalid for use in DeepDFA. This file caches a manifest of which CFG files are valid.
    - `id`: the id of the graph being referenced.
    - `valid`: True if the graph is valid, False otherwise.
- Preprocessed files in `DDFA/storage/processed`:
  - `DDFA/storage/processed/bigvul/before`: The CFGs produced by Joern. The CFGs are organized as a collection of files where `<id>` is the numeric ID of the  corresponding dataset example:
    - `DDFA/storage/processed/bigvul/before/<id>.c`: Source code of the example.
    - `DDFA/storage/processed/bigvul/before/<id>.c.nodes.json`: Serialized list of AST nodes in the example's CFG format.
      - Each object in the array contains information about one node.
      - `id`: a number uniquely identifying the node within a graph.
      - `_label`: the type of the node, assigned by Joern.
      - `columnNumber`: the column number this node occurs on.
      - `lineNumber`: the line number this node occurs on.
      - `order`: the left-to-right order of the node, relative to any sibling nodes.
      - `code`: the source code of the node.
      - `name`: the identifier, if any, referenced in this node.
      - `typeFullName`: the type declared in the node.
    - `DDFA/storage/processed/bigvul/before/<id>.c.edges.json`: Serialized list of edges in the example's CFG format.
      - Each object in the array contains information about one directed edge.
      - Index `0`: the id of the in-node of this edge.
      - Index `1`: the id of the out-node of this edge.
      - Index `2`: the type of the edge, e.g. AST, CFG, ...
      - Index `3`: dataflow information attached to this edge.
    - `DDFA/storage/processed/bigvul/before/<id>.c.cpg.bin`: Saved CFG generated by Joern, in Joern's binary format.
    - To unpack the above CFGs, unzip `before.zip` in `DDFA/storage/processed/bigvul`.
  - `DDFA/storage/processed/bigvul/eval/statement_labels.pkl`: A list of which lines are dependent on the lines added in Big-Vul, used to compute the line-level vulnerability labels.
  - `DDFA/storage/processed/bigvul/nodes.csv`: A CSV format of the CFG nodes in `DDFA/storage/processed/bigvul/before`.
  - `DDFA/storage/processed/bigvul/edges.csv`: A CSV format of the CFG edges in `DDFA/storage/processed/bigvul/before`.
  - `DDFA/storage/processed/bigvul/graphs.bin`: A single file containing the DGL-format graphs, produced by `DDFA/sastvd/scripts/dbize_graphs.py` and loaded by `DDFA/sastvd/linevd/graphmogrifier.py`.
  - `DDFA/storage/processed/bigvul/nodes_feat__ABS_DATAFLOW_<settings>.csv`: Abstract dataflow information for one feature. Some of these files will be combined to form the optimal feature set.
    - Each row contains information about the abstract dataflow features attached to a node.
    - `node_id`: the id of the node which this row is referencing.
    - `graph_id`: the id of the graph which this row is referencing.
    - `_ABS_DATAFLOW_<feature>_all_limitall_<all-limit>_limitsubkeys_<subkey-limit>`: the index of the abstract dataflow feature.
      - `feature`: which of the four features is being referenced; one of `api`, `datatype`, `literal`, or `operator`.
      - `all-limit`: the maximum number of indices in the abstract dataflow embedding; all less-frequent items are given the `unknown` index as a placeholder.
      - `subkey-limit`: the maximum number of indices in any individual abstract dataflow feature (i.e. subkey); all less-frequent items are given the `unknown` index as a placeholder.
  - `DDFA/storage/processed/bigvul/abstract_dataflow_hash_api_datatype_literal_operator.csv`: The selected abstract dataflow features with the optimal settings, which will be used for DeepDFA.
    - Each row contains information about the abstract dataflow embedding attached to a node.
    - `node_id`: the id of the node which this row is referencing.
    - `graph_id`: the id of the graph which this row is referencing.
    - `hash`: the combination of abstract dataflow features occurring in this node.
  - To unpack the above preprocessed and cache files, unzip `preprocessed_data.zip` in the repository root.

# Setup

- Hardware: We ran the experiments on an AMD Ryzen 5 1600 3.2 GHz processor with 48GB of RAM and an Nvidia 3090 GPU with 24GB of GPU memory.
- Software:
  - Linux operating system (we tested on Ubuntu version 22.04).
  - Anaconda3 (we tested on version 23.11.0).
  - CUDA and CUDA toolkit (we tested on version 11.8).

We also provide a Docker container which fulfills the software requirements. To use the container with CUDA, please ensure you have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). We provided a script to run without CUDA, and without installing NVIDIA Container Toolkit, if desired.

# Usage

Use these scripts to run the main performance experiments from the paper (Table 3b).

## Docker

These instructions start up a Docker container with the requisite requirements.

```bash
# Download the code
git clone https://github.com/ISU-PAAL/DeepDFA
cd DeepDFA

# Start the Docker container with an interactive shell; requires NVIDIA Container Toolkit to be installed.
bash scripts/docker_run.sh # may take up to 10 minutes to download the Docker container
# Optional: run without CUDA; this removes the requirement for NVIDIA Container Toolkit.
bash scripts/docker_run_cpu.sh # may take up to 10 minutes to download the Docker container

# Now inside the Docker container shell

# Download the data
bash scripts/download_all.sh # takes about 30 minutes
# Run the main experiments
bash scripts/performance_evaluation.sh # Takes about 20 hours to run all experiments; see below for individual experiments
# Optional: run without CUDA; this removes the requirement for NVIDIA Container Toolkit.
bash scripts/performance_evaluation_cpu.sh
```

## Simple usage

We provided wrapper scripts to execute the various steps.

```bash
# Download the code
git clone https://github.com/ISU-PAAL/DeepDFA
cd DeepDFA

# Setup the environment
. scripts/setup_environment.sh # may take up to 20 minutes to download the packages
# Download the data
bash scripts/download_all.sh
# Run the main experiments
bash scripts/performance_evaluation.sh # Takes about 20 hours to run all experiments; see below for individual experiments
```

## Get the code

```bash
git clone https://github.com/ISU-PAAL/DeepDFA
cd DeepDFA
```

## Set up dependencies

```bash
# In repository root directory
. scripts/setup_environment.sh
# Optional: install joern and add it to the executable path. Takes about 10 minutes
bash $(dirname $0)/install_joern.sh
export PATH="$PWD/joern/joern-cli:$PATH"
```

## Unpack data

Our code assumes that the datasets and preprocessed data files are placed in the following locations:
- Unzip `MSR_data_cleaned.zip` in `DDFA/storage/external/` (1.44 GB, 10.79 GB unzipped).
- Unzip `MSR_LineVul.zip` in `LineVul/data/MSR/` (1.72 GB, 10.96 GB unzipped).
- Unzip `preprocessed_data.zip` in the repository root (1.67 GB, 8.87 GB unzipped).
- Unzip `before.zip` in `DDFA/storage/processed/bigvul` (3.38 GB, 14.70 GB unzipped).

The following script downloads and unpacks all the data to the proper locations.

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
bash scripts/train.sh --seed_everything 1 # takes about 30 minutes
```

## Train LineVul baseline or DeepDFA+LineVul

These scripts report performance of the LineVul and DeepDFA+LineVul models, comparable to Table 3b in our paper.

```bash
cd LineVul/linevul
# Train LineVul
bash scripts/msr_train_linevul.sh 1 MSR # takes about 10 hours
# Train DeepDFA+LineVul
bash scripts/msr_train_combined.sh 1 MSR # takes about 10 hours
```

## Run end-to-end processing

The above scripts use the preprocessed data included in our data archive, for ease of replicability. The instructions below show how to run the code end-to-end.

### On sample data

The current prototype scripts take some time to process data into the format for our dataset, so we provide instructions how to do it with sample mode or full data mode.

```bash
cd DDFA
bash scripts/preprocess.sh --sample # may take several hours. Requires Joern to be installed
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
bash scripts/preprocess.sh # may take several hours. Requires Joern to be installed
# Train DeepDFA
bash scripts/train.sh --seed_everything 1
```

# Extended experiment instructions

This section contains extended instructions for running additional experiments reported in our paper.
For the experiments below, see our extended data package at https://doi.org/10.6084/m9.figshare.21225413.v1, file `data.zip`.

## DeepDFA

We forked this portion of our implementation from [LineVD](https://github.com/davidhin/linevd).
We used their code to generate CFGs with [Joern v1.1.1072](https://joern.io/) and load the dataset.

Scripts for running the different experiments:

```bash
bash scripts/train.sh --seed_everything 1                   # train on MSR, see Table 3b
bash scripts/run_profiling.sh <checkpoint_from_training>    # run profiling on trained checkpoint, see Table 5
bash scripts/run_cross_project.sh                           # train on mixed-project, evaluate on mixed- and cross-project, see Table 7
```

The coverage of the abstract dataflow embedding (running with `--analyze_dataset`) is logged in `logs/1.Effectiveness/DDFA/analyze_dataset.log`.

## LineVul and DeepDFA+LineVul

We forked this portion of our implementation from [LineVul](https://github.com/awsm-research/LineVul).
We used their code to evaluate LineVul and combine with DDFA.

```bash
# run DDFA preprocessing before running any *_combined.sh

cd LineVul/linevul
# MSR
bash scripts/msr_train_linevul.sh 1 MSR                 # original LineVul model (without DeepDFA), see Table 3b
bash scripts/msr_train_combined.sh 1 MSR                # LineVul + DeepDFA, see Table 3b
# cross project, see Table 7
bash scripts/cross_project_train_linevul.sh  1 cross_project/fold_0_holdout
bash scripts/cross_project_eval_linevul.sh   1 saved_models/cross_project-fold_0_dataset/checkpoint-best-f1/1_linevul.bin  cross_project/fold_0_holdout
bash scripts/cross_project_train_combined.sh 1 cross_project/fold_0_holdout
bash scripts/cross_project_eval_combined.sh  1 saved_models/cross_project-fold_0_dataset/checkpoint-best-f1/1_combined.bin cross_project/fold_0_holdout
```

`missing_ids.txt` is a list of the 7% of the dataset examples which could not be parsed by DDFA.

## Reporting profiling records

Use `scripts/report_profiling.py`. For example:

```
python scripts/report_profiling.py --profiledata logs/2.Efficiency/profiling/DDFA/gpu_bs1/profiledata.jsonl
gflops: 763.31798 average: 0.04079514617070173
gmacs: 763.31798 average: 0.04079514617070173
```

For LineVul and LineVul+DDFA, GMACs are printed in `profile.log` (consider the first occurrence of `XXX GMACs`).

## Ablation study, see Table 9

The scripts for training ablation models and evaluating them on DbgBench are in `ablation_study/scripts`.
The scripts, notebooks, and model outputs for extracting performance metrics are in `ablation_study/logs_bigvul` and `ablation_study/logs_dbgbench`.

## CodeT5, see Table 3b

The scripts for training CodeT5 and DeepDFA+CodeT5 are in `CodeT5/code/CodeT5/sh`.
The logs of running the scripts are in `CodeT5/logs/*/train.log`.

## UniXcoder, see Table 3b

The scripts for training UniXcoder and DeepDFA+UniXcoder are in `UniXcoder/scripts`.
This includes an updated script, `linevul_main.py`, for running the LineVul model with a UniXcoder backbone and can be placed into the `LineVul/linevul` source directory to run with the code in the main data package.
The logs of running the scripts are in the various directories `logs_*`.
Notebooks are included to summarize model performance in `logs_size` and `logs_crossproject` and a script is included for DbgBench in `logs_dbgbench`.

## Statistical tests, see Table 4

The scripts and model outputs for running statistical tests on LLM vs. DeepDFA+LLM are in `statistical_test`.
Please see `linevul.ipynb` to run LineVul vs. DeepDFA+LineVul, and
Please see `unixcoder.ipynb` to run UniXcoder vs. DeepDFA+UniXcoder.

# Citation

If you used our code in your research, please consider citing our paper:

> Benjamin Steenhoek, Hongyang Gao, and Wei Le. 2024. Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection. In 2024 IEEE/ACM 46th International Conference on Software Engineering (ICSE ’24), April 14–20, 2024, Lisbon, Portugal. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3597503.3623345
