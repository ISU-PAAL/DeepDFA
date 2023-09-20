testfile=$1
weight_path=$2
seed=$3
model=$4

python linevul_main.py \
  --model_name=$weight_path \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/$model-base \
  --model_name_or_path=microsoft/$model-base \
  --eval_export \
  --do_test --dbgbench_ddfa \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=all \
  --seed $seed \
  --test_data_file=$testfile \
  --block_size 512 \
  --eval_batch_size 4 2>&1 | tee "eval_export_dbgbench_${model}_$(echo $testfile| sed s@/@-@g)_$(echo $weight_path | sed s@/@-@g)_$seed.log"
