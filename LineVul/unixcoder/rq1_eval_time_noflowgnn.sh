dataset=$1

python linevul_main.py \
  --output_dir=saved_models_uxc--no_flowgnn \
  --model_name=0 \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base \
  --model_name_or_path=microsoft/unixcoder-base \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 1 --time --no_flowgnn 2>&1 | tee "eval_time_uxc_${1}_$(echo $2 | sed 's@/@-@g').log"
