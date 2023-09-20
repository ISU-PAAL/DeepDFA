import numpy as np
import pandas as pd
import sastvd as svd
import torch as th
import tqdm
import functools

from dgl.data.utils import load_graphs
import dgl


import logging
logger = logging.getLogger(__name__)

allfeats = [
    "api", "datatype", "literal", "operator",
]

@functools.cache
def get_nodes_df(dsname, sample_mode, feat, concat_all_absdf=False, load_features=True):
    sample_text = "_sample" if sample_mode else ""
    cols = ["Unnamed: 0", "graph_id", "node_id", "dgl_id", "vuln", "code", "_label"]
    nodes = pd.read_csv(svd.processed_dir() / dsname / f"nodes{sample_text}.csv", index_col=0, usecols=cols, dtype={"code": str}, na_values = [])
    nodes = nodes.reset_index(drop=True)
    nodes.code = nodes.code.astype(str)
    split = "fixed"
    if load_features:
        if feat is not None:
            nodes = pd.merge(nodes, pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_{feat}_{split}{sample_text}.csv", index_col=0), how="left", on=["graph_id", "node_id"])
        if concat_all_absdf:
            prefix = "_ABS_DATAFLOW_"
            rest = feat[feat.index("_all"):]
            for otherfeat in allfeats:
                otherdf = pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_{prefix}{otherfeat}{rest}_{split}{sample_text}.csv", index_col=0)
                otherdf = otherdf.rename(columns={next(c for c in otherdf.columns if c.startswith("_ABS_DATAFLOW")): f"_ABS_DATAFLOW_{otherfeat}"})
                # print("other df", otherfeat)
                # print(otherdf)
                nodes = pd.merge(nodes, otherdf, how="left", on=["graph_id", "node_id"])

    return nodes


@functools.cache
def get_df_df(dsname, sample_mode):
    sample_text = "_sample" if sample_mode else ""
    df_df = pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_DF{sample_text}.csv", index_col=0)
    df_df = df_df[["graph_id", "node_id", "df_in"]]
    return df_df


@functools.cache
def get_graphs_by_id(dsname, sample_mode):
    sample_text = "_sample" if sample_mode else ""
    graphs, graph_labels = load_graphs(str(svd.processed_dir() / dsname / f"graphs{sample_text}.bin"))
    graphs_by_id = dict(zip(graph_labels["graph_id"].tolist(), graphs))
    return graphs_by_id


def get_graphs(dsname, nodes_df, sample_mode, feat, partition, concat_all_absdf, load_features):
    graphs_by_id = get_graphs_by_id(dsname, sample_mode)
    feats_init = []
    for i in range(len(graphs_by_id)):
        feats_init.append(dict())
    extrafeats_by_id = dict(zip(graphs_by_id.keys(), feats_init))
    if not load_features:
        return graphs_by_id, extrafeats_by_id

    # update graph features
    partition_graphs_by_id = {}
    skipped_df = 0
    node_len = []
    printed = 0
    was_vuln = []
    for graph_id, group in tqdm.tqdm(nodes_df.groupby("graph_id"), f"graphize {partition}"):
        g: dgl.HeteroGraph = graphs_by_id[graph_id]
        g.ndata["_ABS_DATAFLOW"] = th.LongTensor(group[feat].tolist())
        if concat_all_absdf:
            for otherfeat in allfeats:
                g.ndata[f"_ABS_DATAFLOW_{otherfeat}"] = th.LongTensor(group[f"_ABS_DATAFLOW_{otherfeat}"].tolist())

        g.ndata["_VULN"] = th.Tensor(group["vuln"].tolist()).int()
        was_vuln.append(group["vuln"].max().item())

        if printed < 5:
            logger.debug("graph %d: %s\n%s", graph_id, g, g.ndata)
            printed += 1

        partition_graphs_by_id[graph_id] = g
    graphs_by_id = partition_graphs_by_id
    node_len = np.array(node_len)

    logger.info("percentage of vuln graphs: %s", np.average(was_vuln))
    logger.info("percentage of vuln nodes:\n%s", nodes_df.value_counts("vuln", normalize=True))
    logger.info("percentage of graphs with at least 1 vuln:\n%s", nodes_df.groupby("graph_id")["vuln"].agg(lambda g: 1 if g.any() else 0).value_counts(normalize=True))
    logger.info("skipped dataflow: %d", skipped_df)

    return graphs_by_id, extrafeats_by_id
