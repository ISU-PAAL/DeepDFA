#%%
import pandas as pd
import sastvd.helpers.datasets as svdds
import sastvd as svd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true")
parser.add_argument("--dsname", default="bigvul")
args = parser.parse_args()

sample_mode = args.sample
dsname = args.dsname

sample_text = "_sample" if sample_mode else ""
cols = ["Unnamed: 0", "graph_id", "node_id"]
node_dfs = pd.read_csv(svd.processed_dir() / dsname / f"nodes{sample_text}.csv", index_col=0, usecols=cols)
node_dfs

#%%
for limitall in [1, 10, 100, 500, 1000, 5000, 10000]:
    for sfeat in ["datatype", "api", "literal", "operator"]:
        my_node_df = node_dfs.copy()
        limitsubkeys = limitall
        split = "fixed"
        seed = 0
        feat = f"_ABS_DATAFLOW_{sfeat}_all_limitall_{limitall}_limitsubkeys_{limitsubkeys}"
        dst_file = svd.processed_dir() / dsname / f"nodes_feat_{feat}_{split}{sample_text}.csv"
        print("processing", feat, "to", dst_file)

        abs_df, abs_df_hashes = svdds.abs_dataflow(feat, dsname, sample_mode, split=split, seed=seed)
        all_hash_idxs = abs_df_hashes["all"]
        all_hashes = abs_df.set_index(["graph_id", "node_id"])["hash.all"]

        def get_hash_idx(row):
            _hash = all_hashes.get((row["graph_id"], row["node_id"]), None)
            if _hash is None:
                # nid not in abstract features - not definition
                return 0
            else:
                # if None, then nid maps to UNKNOWN token
                return all_hash_idxs.get(_hash, all_hash_idxs[None]) + 1
        my_node_df[feat] = svd.dfmp(my_node_df, get_hash_idx, ["graph_id", "node_id"])
        my_node_df.to_csv(dst_file)
        print(dst_file, "saved")
