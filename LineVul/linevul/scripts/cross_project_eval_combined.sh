seed=$1
weight_subset=$2
subset=$3

python linevul_main.py \
  --model_name=$weight_subset \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --test_data_file=../data/$subset/holdout.csv \
  --block_size 512 \
  --eval_batch_size 4 --seed $seed 2>&1 | tee "eval_$(echo $subset | sed s@/@-@g)_$(echo $weight_subset | sed s@/@-@g)_$seed.log"
