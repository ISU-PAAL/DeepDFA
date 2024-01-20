#!/bin/bash
set -e

# Train DeepDFA
(cd DDFA; bash scripts/train.sh --seed_everything 1 --trainer.gpus 0)
# Train LineVul
(cd LineVul/linevul; bash scripts/msr_train_linevul.sh 1 MSR)
# Train DeepDFA+LineVul
(cd LineVul/linevul; bash scripts/msr_train_combined.sh 1 MSR)
