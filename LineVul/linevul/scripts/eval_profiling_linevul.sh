dataset=$1

python linevul_main.py \
  --model_name=$2 \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 16 --profile --no_flowgnn 2>&1 | tee "eval_${1}_$(echo $2 | sed 's@/@-@g').log"
