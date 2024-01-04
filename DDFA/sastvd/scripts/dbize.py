#%%
import sastvd.helpers.datasets as svdds
import sastvd as svd
import sastvd.helpers.evaluate as ivde
from sastvd.linevd.utils import feature_extraction

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true")
parser.add_argument("--dsname", default="bigvul")
args = parser.parse_args()

sample_mode = args.sample
dsname = args.dsname

df = svdds.ds(dsname, cache=True, sample=sample_mode)
df = svdds.ds_filter(
    df,
    dsname,
    check_file=True,
    check_valid=True,
    vulonly=False,
    load_code=False,
    sample=-1,
    sample_mode=sample_mode,
)
print(df)

#%%
if dsname == "bigvul":
    graph_type = "cfg"
    dep_add_lines = ivde.get_dep_add_lines_bigvul("bigvul", sample=sample_mode)
    dep_add_lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in dep_add_lines.items()}

    def get_vuln(lineno, _id, dep_add_lines):
        if _id in dep_add_lines and lineno in dep_add_lines[_id]:
            return 1
        else:
            return 0

    def graph_features(_id):
        itempath = svdds.itempath(_id)
        n, e = feature_extraction(itempath,
            graph_type=graph_type,
            return_nodes=True,
            return_edges=True,
            group=False,
            )
        n["vuln"] = n.lineNumber.apply(get_vuln, _id=_id, dep_add_lines=dep_add_lines)
        n = n.drop(columns=["id"])
        n = n.reset_index().rename(columns={"index": "dgl_id"})
        n["graph_id"] = _id
        e["graph_id"] = _id
        n = n.reset_index(drop=True)
        e = e.reset_index(drop=True)
        return n, e

    node_dfs, edge_dfs = zip(*svd.dfmp(df, graph_features, "id"))
elif dsname == "devign":
    graph_type = "cfg"

    def graph_features(row):
        _id = row["id"]
        target = row["target"]
        itempath = svdds.itempath(_id, dsname)
        n, e = feature_extraction(itempath,
            graph_type=graph_type,
            return_nodes=True,
            return_edges=True,
            group=False,
            )
        n["vuln"] = target
        n = n.drop(columns=["id"])
        n = n.reset_index().rename(columns={"index": "dgl_id"})
        n["graph_id"] = _id
        e["graph_id"] = _id
        n = n.reset_index(drop=True)
        e = e.reset_index(drop=True)
        return n, e

    node_dfs, edge_dfs = zip(*svd.dfmp(df, graph_features, ["id", "target"]))

#%%
node_dfs[0]

#%%
edge_dfs[0]

#%%
print(node_dfs[0])
print(edge_dfs[0])

#%%
import pandas as pd
node_dfs = pd.concat(node_dfs, ignore_index=True)
edge_dfs = pd.concat(edge_dfs, ignore_index=True)

#%%
print("percentage of vuln nodes:", node_dfs.value_counts("vuln", normalize=True), sep="\n")
print("percentage of graphs with at least 1 vuln:", node_dfs.groupby("graph_id")["vuln"].agg(lambda g: 1 if g.any() else 0).value_counts(normalize=True), sep="\n")

#%%
sample_text = "_sample" if sample_mode else ""
node_dfs.to_csv(svd.processed_dir() / dsname / f"nodes{sample_text}.csv")
edge_dfs.to_csv(svd.processed_dir() / dsname / f"edges{sample_text}.csv")

print("done")
