testfile=$1
weight_path=$2
seed=$3

python linevul_main.py \
  --model_name=$weight_path \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base \
  --model_name_or_path=microsoft/unixcoder-base \
  --eval_export \
  --do_test --no_flowgnn --really_no_flowgnn \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=all \
  --seed $seed \
  --test_data_file=$testfile \
  --block_size 512 \
  --eval_batch_size 4 2>&1 | tee "eval_export_dbgbench_$(echo $testfile| sed s@/@-@g)_$(echo $weight_path | sed s@/@-@g)_$seed.log"
