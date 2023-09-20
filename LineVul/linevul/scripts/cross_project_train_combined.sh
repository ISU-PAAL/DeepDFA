seed=$1
subset=$2
subset_slug="$(echo $subset | sed s@/@-@g)"

python linevul_main.py \
  --model_name=${seed}_combined.bin \
  --output_dir=./saved_models/$subset_slug \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=../data/$subset/train.csv \
  --eval_data_file=../data/$subset/valid.csv \
  --test_data_file=../data/$subset/test.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed 2>&1 | tee "train_subset_$(echo $subset | sed s@/@-@g).log"
