# Reproduction package for "Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection"

We included the code to reproduce our experiments on DeepDFA, LineVul, LineVul+DeepDFA, UniXcoder, and CodeT5, to be published at ICSE 2024.
This data package includes the code and data that was checked during peer review and was prepared on September 20, 2023.
Updates to the code will be hosted on the GitHub repo.

Links:
* Paper: https://doi.org/10.1145/3597503.3623345
* Arxiv preprint: https://arxiv.org/abs/2212.08108
* Data package: https://doi.org/10.6084/m9.figshare.21225413
* GitHub repo: https://github.com/ISU-PAAL/DeepDFA

If you used our code in your research, please consider citing our paper:

> Benjamin Steenhoek, Hongyang Gao, and Wei Le. 2024. Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection. In 2024 IEEE/ACM 46th International Conference on Software Engineering (ICSE ’24), April 14–20, 2024, Lisbon, Portugal. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3597503.3623345

Here is how the files are organized.

```
├── DDFA: the code for the DeepDFA model.
├── CodeT5: the code for the CodeT5 and CodeT5+DeepDFA models.
├── LineVul: the code for the LineVul and LineVul+DeepDFA models.
├── logs: the logs output during our training and evaluation process.
├── models: the saved checkpoints of the models we trained and evaluated. This excludes UniXcoder and CodeT5 checkpoints for lack of space and may be available upon request.
├── scripts: miscellaneous scripts we used to report the results.
└── README.md
```

We ran the experiments on an AMD Ryzen 5 1600 3.2 GHz processor with 48GB of RAM and an Nvidia 3090 GPU with 24GB of GPU memory and CUDA 11.3.
To reduce the size of the package, we included 1 seed/fold of the LineVul model checkpoints for each experiment. The other checkpoints will be made available upon request.

## DeepDFA code

Forked from [LineVD](https://github.com/davidhin/linevd).
We used their code to generate CFGs with [Joern v1.1.1072](https://joern.io/) and load the dataset.

First, run setup:

```bash
cd DDFA
# setup environment
pip install virtualenv
virtualenv -p 3.10 venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"

# unpack CFG dataset
cd storage/processed/bigvul
unzip before.zip
cd -

# preprocess (data are already available)
# sbatch --wait scripts/run_prepare.sh        # or use bash if not on slurm
# sbatch --wait scripts/run_getgraphs.sh      # or use bash if not on slurm
# bash scripts/run_dbize.sh
# bash scripts/run_abstract_dataflow.sh

bash scripts/train.sh --seed_everything 1                   # train on MSR
bash scripts/run_profiling.sh <checkpoint_from_training>    # run profiling on trained checkpoint
bash scripts/run_cross_project.sh                           # train on mixed-project, evaluate on mixed- and cross-project
```

The coverage of the abstract dataflow embedding (running with `--analyze_dataset`) is logged in `logs/1.Effectiveness/DDFA/analyze_dataset.log`.

## LineVul code

Forked from [LineVul](https://github.com/awsm-research/LineVul).
We used their code to evaluate LineVul and combine with DDFA.

First, run setup:
```bash
cd LineVul
# setup environment
pip install virtualenv
virtualenv -p 3.10 venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/../DDFA:$PYTHONPATH"

# run DDFA preprocessing before running any *_combined.sh

cd linevul
# MSR
bash scripts/msr_train_linevul.sh 1 MSR                 # original LineVul model (without DeepDFA)
bash scripts/msr_train_combined.sh 1 MSR                # LineVul + DeepDFA
# cross project
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

## Ablation study
The scripts for training ablation models and evaluating them on DbgBench are in `ablation_study/scripts`.
The scripts, notebooks, and model outputs for extracting performance metrics are in `ablation_study/logs_bigvul` and `ablation_study/logs_dbgbench`.

## CodeT5
The scripts for training CodeT5 and DeepDFA+CodeT5 are in `CodeT5/code/CodeT5/sh`.
The logs of running the scripts are in `CodeT5/logs/*/train.log`.

## UniXcoder
The scripts for training UniXcoder and DeepDFA+UniXcoder are in `UniXcoder/scripts`.
This includes an updated script, `linevul_main.py`, for running the LineVul model with a UniXcoder backbone and can be placed into the `LineVul/linevul` source directory to run with the code in the main data package.
The logs of running the scripts are in the various directories `logs_*`.
Notebooks are included to summarize model performance in `logs_size` and `logs_crossproject` and a script is included for DbgBench in `logs_dbgbench`.

## Statistical tests
The scripts and model outputs for running statistical tests on LLM vs. DeepDFA+LLM are in `statistical_test`.
Please see `linevul.ipynb` to run LineVul vs. DeepDFA+LineVul, and
Please see `unixcoder.ipynb` to run UniXcoder vs. DeepDFA+UniXcoder.
