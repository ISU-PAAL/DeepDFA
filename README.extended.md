# Extended README

This file contains extended instructions for running additional experiments reported in our paper.

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

# Data package

For the experiments below, see our data package at https://doi.org/10.6084/m9.figshare.21225413.v1, file `data.zip`.

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
