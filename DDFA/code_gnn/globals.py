import random
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything

project_root_dir = Path(__file__).parent.parent
ml_data_dir = project_root_dir / 'data'
test_data_dir = project_root_dir / 'test_data'
wv_cache_path = project_root_dir / 'wv_cache'


all_models = ['devign', 'flow_gnn', 'flow_gnn_only', 'random']
all_datasets = ['devign', 'combined', 'MSR', 'CVEFixes', 'D2A', 'combined_new', 'SARD']
all_aggregate_functions = ['sum', 'mean', 'max', 'bitwise_union_relu', 'bitwise_union_simple']


def get_cache_filename(dataset_name, model_name, node_limit, graph_limit):
    return f'cache_{dataset_name}_{model_name}_{node_limit}_{graph_limit}.pkl'


def test_data_dir_exists():
    assert ml_data_dir.is_dir()
    print(f'ml_data_dir={ml_data_dir}')
    assert test_data_dir.is_dir()
    print(f'test_data_dir={test_data_dir}')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    seed_everything(seed, workers=True)
