#%%
import pandas as pd
import sastvd as svd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true")
parser.add_argument("--dsname", default="bigvul")
args = parser.parse_args()

sample_mode = args.sample
dsname = args.dsname

sample_text = "_sample" if sample_mode else ""
cols = ["Unnamed: 0", "graph_id", "innode", "outnode"]
edge_dfs = pd.read_csv(svd.processed_dir() / dsname / f"edges{sample_text}.csv", index_col=0, usecols=cols)
edge_dfs

#%%
import dgl
graphs = []
graph_ids = []
for graph_id, group in edge_dfs.groupby("graph_id"):
    g = dgl.graph((group["innode"].tolist(), group["outnode"].tolist()))
    g = dgl.add_self_loop(g)
    graphs.append(g)
    graph_ids.append(graph_id)

#%%
from dgl.data.utils import save_graphs
import torch as th
print({"graph_id": th.LongTensor(graph_ids)})
save_graphs(str(svd.processed_dir() / dsname / f"graphs{sample_text}.bin"), graphs, {"graph_id": th.LongTensor(graph_ids)})

print("done")
