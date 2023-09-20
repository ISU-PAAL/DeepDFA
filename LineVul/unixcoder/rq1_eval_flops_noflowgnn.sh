dataset=$1

exec python linevul_main.py \
  --output_dir=saved_models_uxc--no_flowgnn \
  --model_name=0 \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base \
  --model_name_or_path=microsoft/unixcoder-base \
  --do_test \
  --test_data_file=../data/$dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 1 --profile --no_flowgnn 2>&1 | tee "eval_flops_uxc_${1}_$(echo $2 | sed 's@/@-@g').log"
