#!/bin/bash

set -e

bash scripts/run_prepare.sh $@
bash scripts/run_getgraphs.sh $@ # Make sure Joern is installed!
bash scripts/run_dbize.sh $@
bash scripts/run_abstract_dataflow.sh $@
bash scripts/run_absdf.sh $@
