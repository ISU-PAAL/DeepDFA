for s in 0 1 2
do
    bash eval_export.sh \
        ../data/additional_experiment_data/additional_experiments/dbgbench/dbgbench_data_code.csv \
        saved_models_uxc--no_flowgnn/checkpoint-best-f1/$s $s
done
