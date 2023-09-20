# for s in 0 1
# for s in 2
# do
#     bash eval_export_combined.sh \
#         ../data/additional_experiment_data/additional_experiments/dbgbench/dbgbench_data_code.csv \
#         saved_models_uxc/checkpoint-best-f1/$s $s unixcoder
# done
# for s in 0 1
for s in 2
do
    bash eval_export_combined.sh \
        ../data/additional_experiment_data/additional_experiments/dbgbench/dbgbench_data_code.csv \
        saved_models/checkpoint-best-f1/$s $s codebert
done
