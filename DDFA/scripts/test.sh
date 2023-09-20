ckpt_path="$1"
shift
python code_gnn/main_cli.py test --config configs/config_bigvul.yaml --config configs/config_ggnn.yaml --ckpt_path "$ckpt_path" $@
