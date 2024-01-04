import functools
import os
import re

import numpy as np
import pandas as pd
import sastvd as svd
from glob import glob
from pathlib import Path
import json
import traceback
import sastvd.helpers.git as svdg
import sastvd.helpers.joern as svdj
import logging

logger = logging.getLogger(__name__)


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def devign(cache=True, sample=False):
    """
    Read devign dataset from JSON
    """

    savefile = (
        svd.get_dir(svd.cache_dir() / "minimal_datasets")
        / f"minimal_devign{'_sample' if sample else ''}.pq"
    )
    if cache:
        try:
            df = pd.read_parquet(savefile, engine="fastparquet").dropna()

            return df
        except FileNotFoundError:
            logger.info(f"file {savefile} not found, loading from source")
        except Exception:
            logger.exception("devign exception, loading from source")

    filename = "function.json"
    df = pd.read_json(svd.external_dir() / filename,)
    df = df.rename_axis("id").reset_index()
    df["dataset"] = "devign"

    # Remove comments
    df["before"] = svd.dfmp(df, remove_comments, "func", cs=500)
    df["before"] = df["before"].apply(lambda c: c.replace("\n\n", "\n"))

    # Remove functions with abnormal ending (no } or ;)
    df = df[
        ~df.apply(
            lambda x: x.before.strip()[-1] != "}"
            and x.before.strip()[-1] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    df = df[~df.before.apply(lambda x: x[-2:] == ");")]
    df["vul"] = df["target"]

    # # Remove samples with mod_prop > 0.5
    # dfv["mod_prop"] = dfv.apply(
    #     lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    # )
    # dfv = dfv.sort_values("mod_prop", ascending=0)
    # dfv = dfv[dfv.mod_prop < 0.7]
    # # Remove functions that are too short
    # dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]

    if sample:
        df = df.head(50)

    minimal_cols = [
        "id",
        "dataset",
        "before",
        "target",
        "vul",
    ]
    df[minimal_cols].to_parquet(
        savefile,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    return df


def mutated(subdataset, cache=True, sample=False):
    """
    Read mutated dataset from JSON
    """

    df = bigvul(cache=cache, sample=sample)
    df = df.drop(columns=["dataset", "before"])
    fp = svd.external_dir() / "mutated" / f"c_{subdataset.replace('_flip', '')}.jsonl"
    # print("loading", fp)
    mutated = pd.read_json(fp, lines=True)
    if "flip" in subdataset:
        mutated = mutated.rename(columns={"source": "before"}).drop(columns=["target"])
    else:
        mutated = mutated.rename(columns={"target": "before"}).drop(columns=["source"])
    df = pd.merge(df, mutated, left_on="id", right_on="idx",
        # how="left", # 
        how="inner", # include only examples with mutated code
    )
    df["dataset"] = f"mutated_{subdataset}"
    df = df.drop(columns=["after", "added", "removed", "diff"])

    return df


def ds(dsname, cache=True, sample=False):
    if dsname == "bigvul":
        return bigvul(cache=cache, sample=sample)
    elif dsname == "devign":
        return devign(cache=cache, sample=sample)
    elif "mutated" in dsname:
        subdataset = dsname.split("_", maxsplit=1)[1]
        return mutated(subdataset, cache=cache, sample=sample)


def bigvul(cache=True, sample=False):
    """
    Read BigVul dataset from CSV
    """

    savefile = (
        svd.get_dir(svd.cache_dir() / "minimal_datasets")
        / f"minimal_bigvul{'_sample' if sample else ''}.pq"
    )
    if cache:
        try:
            df = pd.read_parquet(savefile, engine="fastparquet").dropna()

            return df
        except FileNotFoundError:
            logger.info(f"file {savefile} not found, loading from source")
        except Exception:
            logger.exception("bigvul exception, loading from source")

    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(
        svd.external_dir() / filename,
        parse_dates=["Publish Date", "Update Date"],
        dtype={
            "commit_id": str,
            "del_lines": int,
            "file_name": str,
            "lang": str,
            "lines_after": str,
            "lines_before": str,
            "Unnamed: 0": int,
            "Access Gained": str,
            "Attack Origin": str,
            "Authentication Required": str,
            "Availability": str,
            "CVE ID": str,
            "CVE Page": str,
            "CWE ID": str,
            "Complexity": str,
            "Confidentiality": str,
            "Integrity": str,
            "Known Exploits": str,
            "Score": float,
            "Summary": str,
            "Vulnerability Classification": str,
            "add_lines": int,
            "codeLink": str,
            "commit_message": str,
            "files_changed": str,
            "func_after": str,
            "func_before": str,
            "parentID": str,
            "patch": str,
            "project": str,
            "project_after": str,
            "project_before": str,
            "vul": int,
            "vul_func_with_fix": str,
        },
    )
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # Save codediffs
    svd.dfmp(
        df,
        svdg._c2dhelper,
        columns=["func_before", "func_after", "id", "dataset"],
        ordr=False,
        cs=300,
    )

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # POST PROCESSING
    dfv = df[df.vul == 1]
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # Remove functions with abnormal ending (no } or ;)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # Remove samples with mod_prop > 0.5
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # Filter by post-processing filtering
    keep_vuln = set(dfv["id"].tolist())
    df = df[(df.vul == 0) | (df["id"].isin(keep_vuln))].copy()

    minimal_cols = [
        "id",
        "before",
        "after",
        "removed",
        "added",
        "diff",
        "vul",
        "dataset",
    ]
    df[minimal_cols].to_parquet(
        savefile,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    df[
        [
            "id",
            "commit_id",
            "vul",
            "codeLink",
            "commit_id",
            "parentID",
            "CVE ID",
            "CVE Page",
            "CWE ID",
            "Publish Date",
            "Update Date",
            "file_name",
            "files_changed",
            "lang",
            "project",
            "project_after",
            "project_before",
            "add_lines",
            "del_lines",
        ]
    ].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df


def check_validity(_id, dsname, assert_no_exception=True, assert_line_number=False, assert_reaching_def=False):
    """Check whether sample with id=_id can be loaded and has node/edges.

    Example:
    _id = 1320
    with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
        nodes = json.load(f)
    """

    try:
        svdj.get_node_edges(itempath(_id, dsname))
        # check nodes
        with open(str(itempath(_id, dsname)) + ".nodes.json", "r") as f:
            nodes = json.load(f)
        nodes_valid = False
        for n in nodes:
            if "lineNumber" in n.keys():
                nodes_valid = True
                break
        if not nodes_valid:
            logger.warn("valid (%s): no line number", itempath(_id, dsname))
            if assert_line_number:
                return False
        # check edges
        with open(str(itempath(_id, dsname)) + ".edges.json", "r") as f:
            edges = json.load(f)
        edge_set = set([i[2] for i in edges])
        if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
            logger.warn("valid (%s): no dataflow", itempath(_id, dsname))
            if assert_reaching_def:
                return False
    except Exception as E:
        logger.warn("valid (%s): exception\n%s", itempath(_id, dsname), traceback.format_exc())
        if assert_no_exception:
            return False
    return True


def itempath(_id, dsname="bigvul"):
    """Get itempath path from item id. TODO: somehow give itempath of before and after."""
    return svd.processed_dir() / f"{dsname}/before/{_id}.c"


def check_valid_dataflow(_id):
    try:
        d = get_dataflow_output(_id)
        return len(d) > 0
    except Exception:
        traceback.print_exc()
        return False

def bigvul_check_valid_dataflow(df):
    valid = svd.dfmp(df, check_valid_dataflow, "id")
    df = df[valid]
    return df


def ds_filter(
    df,
    dsname,
    check_file=False,
    check_valid=False,
    vulonly=False,
    load_code=False,
    sample=-1,
    sample_mode=False,
    seed=0,
):
    """Filter dataset based on various considerations for training"""

    # Small sample (for debugging):
    if sample > 0:
        df = df.sample(sample, random_state=seed)
    assert len(df) > 0

    # Filter only vulnerable
    if vulonly:
        df = df[df.vul == 1]
    assert len(df) > 0

    # Filter out samples with no parsed file
    if check_file:
        finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / dsname / "before/*nodes*"))
            if not os.path.basename(i).startswith("~")
        ]
        df = df[df.id.isin(finished)]
        logger.debug("check_file %d", len(df))
    assert len(df) > 0

    # Filter out samples with no lineNumber from Joern output
    if check_valid:
        valid_cache = svd.cache_dir() / f"{dsname}_valid_{sample_mode}.csv"
        if valid_cache.exists():
            valid_cache_df = pd.read_csv(valid_cache, index_col=0)
        else:
            valid = svd.dfmp(
                df, functools.partial(check_validity, dsname=dsname), "id", desc="Validate Samples: ", workers=6
            )
            df_id = df.id
            valid_cache_df = pd.DataFrame({"id": df_id, "valid": valid}, index=df.index)
            valid_cache_df.to_csv(valid_cache)
        df = df[df.id.isin(valid_cache_df[valid_cache_df["valid"]].id)]
        logger.debug("check_valid %d", len(df))
    assert len(df) > 0

    # NOTE: drop several columns to save memory
    if not load_code:
        df = df.drop(columns=["before", "after", "removed", "added", "diff"], errors="ignore")
    return df


def bigvul_filter(
    df,
    check_file=False,
    check_valid=False,
    vulonly=False,
    load_code=False,
    sample=-1,
    sample_mode=False,
    seed=0,
):
    return ds_filter(
        df,
        check_file=check_file,
        check_valid=check_valid,
        vulonly=vulonly,
        load_code=load_code,
        sample=sample,
        sample_mode=sample_mode,
        seed=seed,
        dsname="bigvul",
    )


def get_splits_map(dsname):
    logger.debug("loading fixed splits")
    if dsname == "bigvul" or "mutated" in dsname:
        splits = get_linevul_splits()
    if dsname == "devign":
        splits = get_codexglue_splits()
    logger.debug("splits value counts:\n%s", splits.value_counts())
    return splits.to_dict()


def get_linevd_splits_map():
    logger.debug("loading linevd splits")
    splits = pd.read_csv(svd.external_dir() / "bigvul_rand_splits.csv")
    splits = splits.set_index("id")
    logger.debug("splits value counts:\n%s", splits.value_counts())
    return splits.to_dict()


def get_linevul_splits():
    logger.debug("loading linevul splits")
    splits_df = pd.read_csv(svd.external_dir() / "linevul_splits.csv", index_col=0)
    splits = splits_df["split"]
    splits = splits.replace("valid", "val")
    return splits


def get_codexglue_splits():
    splits_df = pd.read_csv(svd.external_dir() / "codexglue_splits.csv")
    splits_df = splits_df.set_index("example_index")
    splits_df["split"] = splits_df["split"].replace("valid", "val")
    splits = splits_df["split"]
    return splits


def get_named_splits_map(split):
    logger.debug("loading %s splits", split)
    splits_df = pd.read_csv(svd.external_dir() / "splits" / f"{split}.csv", index_col=0)
    splits_df = splits_df.set_index("example_index")
    splits = splits_df["split"]
    splits = splits.replace("valid", "val")
    splits = splits.replace("holdout", "test")
    logger.debug("splits value counts:\n%s", splits.value_counts())
    return splits.to_dict()

def ds_partition(
    df, partition, dsname, split="fixed", seed=0,
):
    """Filter to one partition of bigvul and rebalance function-wise"""
    logger.debug(f"ds_partition %d %s %s %d", len(df), dsname, partition, seed)

    if split == "random":
        logger.debug("generating random splits with seed %d", seed)
        splits_map = get_splits_map(dsname)
        df_fixed_splits = df.id.map(splits_map)
        logger.debug("valid splits value counts:\n%s", df_fixed_splits.value_counts())
        df = df[df_fixed_splits != "test"].copy()
        logger.debug("holdout %d test examples from fixed dataset split. dataset len: %d", np.sum(df_fixed_splits == "test"), len(df))

        def get_label(i):
            if i < int(len(df) * 0.1):
                return "val"
            elif i < int(len(df) * 0.2):
                return "test"
            else:
                return "train"

        df["label"] = pd.Series(
            list(map(get_label, range(len(df)))),
            index=np.random.RandomState(seed=seed).permutation(df.index),
        )
        # NOTE: I verified that this always gives the same output for all runs!
        # as long as the input df is the same (should be filtered first e.g. datamodule vs. abs_df)
    elif split == "fixed":
        splits_map = get_splits_map(dsname)
        df["label"] = df.id.map(splits_map)
    elif split == "linevul":
        assert dsname == "bigvul", dsname
        splits_map = get_linevul_splits_map()
        df["label"] = df.id.map(splits_map)
    else:
        assert dsname == "bigvul", dsname
        splits_map = get_named_splits_map(split)
        df["label"] = df.id.map(splits_map)
    logger.debug("dataset value counts\n%s\ndatasethead\n%s", df.value_counts("label"), df.groupby("label").head(5))

    if partition != "all":
        df = df[df.label == partition]
        logger.info(f"partitioned {len(df)}")

    return df

def bigvul_partition(df, partition, split="fixed", seed=0,):
    return ds_partition(df, partition, "bigvul", split, seed)

def test_random():
    df = bigvul()
    df = bigvul_partition(df, seed=42, partition="all", split="random")
    print("TEST 1")
    print(df.value_counts("label"))
    for label, group in df.groupby("label"):
        print(label)
        print(group)

    sdf = bigvul()
    sdf = bigvul_partition(df, seed=42, partition="all", split="random")
    print("TEST 2")
    assert sdf["label"].to_list() == df["label"].to_list()

    odf = bigvul()
    odf = bigvul_partition(odf, seed=53, partition="all", split="random")
    print("TEST 3")
    print(odf.value_counts("label"))
    for label, group in odf.groupby("label"):
        print(label)
        print(group)
        assert len(group) == len(df[df["label"] == label])
        assert group["id"].to_list() != (df[df["label"] == label]["id"]).to_list()
    assert odf["label"].to_list() != df["label"].to_list()


single = {
    "api": False,
    "datatype": True,
    "literal": False,
    "operator": False,
}
all_subkeys = ["api", "datatype", "literal", "operator"]


def parse_limits(feat):
    if "limitsubkeys" in feat:
        start_idx = feat.find("limitsubkeys")+len("limitsubkeys")+1
        end_idx = feat.find("_",start_idx)
        if end_idx == -1:
            end_idx = len(feat)
        limit_subkeys = feat[start_idx:end_idx]
        if limit_subkeys == "None":
            limit_subkeys = None
        else:
            limit_subkeys = int(limit_subkeys)
    else:
        limit_subkeys = 1000
    if "limitall" in feat:
        start_idx = feat.find("limitall")+len("limitall")+1
        end_idx = feat.find("_",start_idx)
        if end_idx == -1:
            end_idx = len(feat)
        limit_all = feat[start_idx:end_idx]
        if limit_all == "None":
            limit_all = None
        else:
            limit_all = int(limit_all)
    else:
        limit_all = 1000
    return limit_subkeys, limit_all

def abs_dataflow(feat, dsname="bigvul", sample=False, split="fixed", seed=0):
    """Load abstract dataflow information"""

    limit_subkeys, limit_all = parse_limits(feat)

    df = ds(dsname, sample=sample)
    df = ds_filter(
        df,
        dsname,
        check_file=True,
        check_valid=True,
        vulonly=False,
        load_code=False,
        sample_mode=sample,
        seed=seed,
    )
    source_df = ds_partition(df, "train", dsname, split=split, seed=seed)

    abs_df_file = (
        svd.processed_dir()
        / dsname / f"abstract_dataflow_hash_api_datatype_literal_operator{'_sample' if sample else ''}.csv"
    )
    if abs_df_file.exists():
        abs_df = pd.read_csv(abs_df_file)
        abs_df_hashes = {}
        abs_df["hash"] = abs_df["hash"].apply(json.loads)
        logger.debug(abs_df)
        # compute concatenated embedding
        for subkey in all_subkeys:
            if subkey in feat:
                logger.debug(f"getting hashes {subkey}")
                hash_name = f"hash.{subkey}"
                abs_df[hash_name] = abs_df["hash"].apply(lambda d: d[subkey])
                if single[subkey]:
                    abs_df[hash_name] = abs_df[hash_name].apply(lambda d: d[0])
                    my_abs_df = abs_df
                else:
                    abs_df[hash_name] = abs_df[hash_name].apply(
                        lambda d: sorted(set(d))
                    )
                    my_abs_df = abs_df.explode(hash_name)
                my_abs_df = my_abs_df[["graph_id", "node_id", "hash", hash_name]]

                hashes = pd.merge(source_df, my_abs_df, left_on="id", right_on="graph_id")[hash_name].dropna()
                # most frequent
                logger.debug(f"min {hashes.value_counts().head(limit_subkeys).min()} {hashes.value_counts().head(limit_subkeys).idxmin()}")
                hashes = (
                    hashes.value_counts()
                    .head(limit_subkeys)
                    .index#.sort_values()
                    .unique()
                    .tolist()
                )
                hashes.insert(0, None)
                # with open("hashes5000", "w") as f:
                #     f.write("\n".join(map(str, hashes)))

                abs_df_hashes[subkey] = {h: i for i, h in enumerate(hashes)}

                logger.debug(f"trained hashes {subkey} {len(abs_df_hashes[subkey])}")

        if "all" in feat:
            def get_all_hash(row):
                h = {}
                for subkey in all_subkeys:
                    if subkey in feat:
                        hash_name = f"hash.{subkey}"
                        hashes = abs_df_hashes[subkey]
                        hash_values = row[hash_name]
                        if "includeunknown" in feat:
                            if single[subkey]:
                                hash_idx = [hash_values]
                            else:
                                hash_idx = hash_values
                        else:
                            if single[subkey]:
                                hash_idx = [
                                    hash_values if hash_values in hashes else "UNKNOWN"
                                ]
                            else:
                                hash_idx = [
                                    hh if hh in hashes else "UNKNOWN"
                                    for hh in hash_values
                                ]
                        h[subkey] = list(sorted(set(hash_idx)))
                return h

            source_df_hashes = pd.merge(source_df, abs_df, left_on="id", right_on="graph_id")
            abs_df["hash.all"] = source_df_hashes.apply(get_all_hash, axis=1).apply(json.dumps)
            hashes = abs_df["hash.all"]
            all_hashes = (
                abs_df["hash.all"]
                .value_counts()
                .head(limit_all)
                .index#.sort_values()
                .unique()
                .tolist()
            )
            all_hashes.insert(0, None)
            # with open("all_hashes5000", "w") as f:
            #     f.write("\n".join(map(str, all_hashes)))
            abs_df_hashes["all"] = {h: i for i, h in enumerate(all_hashes)}

        return abs_df, abs_df_hashes
    else:
        logger.warning("YOU SHOULD RUN `python sastvd/scripts/abstract_dataflow_full.py --stage 2`")


def test_abs():
    abs_df, abs_df_hashes = abs_dataflow(
        feat="_ABS_DATAFLOW_api_datatype_literal_operator", sample=False,
    )
    assert all(not all(abs_df[f"hash.{subkey}"].isna()) for subkey in all_subkeys)
    assert len([c for c in abs_df.columns if "hash." in c]) == len(all_subkeys)
    assert len(abs_df_hashes) == len(all_subkeys)


def test_abs_all():
    for featname in (
        "datatype",
        "literal_operator",
        "api_literal_operator",
        "api_datatype_literal_operator_all",
    ):
        print(featname)
        abs_df, abs_df_hashes = abs_dataflow(
            feat=f"_ABS_DATAFLOW_{featname}_all", sample=False
        )
        vc = abs_df.value_counts("hash.all")
        print(vc)
        print(len(vc.loc[vc > 1].index), "more than 1")
        print(len(vc.loc[vc > 5].index), "more than 5")
        print(len(vc.loc[vc > 100].index), "more than 100")
        print(len(vc.loc[vc > 1000].index), "more than 1000")
        print("min", vc.head(1000).min(), vc.head(1000).idxmin())


def test_abs_all_unk():
    for featname in (
        "datatype",
        "literal_operator",
        "api_literal_operator",
        "api_datatype_literal_operator_all",
    ):
        print(featname)
        abs_df, abs_df_hashes = abs_dataflow(
            feat=f"_ABS_DATAFLOW_{featname}_all_includeunknown", sample=False
        )
        vc = abs_df.value_counts("hash.all")
        print(vc)
        print(len(vc.loc[vc > 1].index), "more than 1")
        print(len(vc.loc[vc > 5].index), "more than 5")
        print(len(vc.loc[vc > 100].index), "more than 100")
        print(len(vc.loc[vc > 1000].index), "more than 1000")
        print("min", vc.head(1000).min(), vc.head(1000).idxmin())


def dataflow_1g(sample=False):
    """Load 1st generation dataflow information"""

    cache_file = svd.processed_dir() / f"bigvul/1g_dataflow_hash_all_{sample}.csv"
    if cache_file.exists():
        df = pd.read_csv(
            cache_file,
            converters={
                "graph_id": int,
                "node_id": int,
                "func": str,
                "gen": str,
                "kill": str,
            },
        )
        df["gen"] = df["gen"].apply(json.loads)
        df["kill"] = df["kill"].apply(json.loads)
        return df
    else:
        logger.warning("YOU SHOULD RUN dataflow_1g.py")


def test_1g():
    print(dataflow_1g(sample=True))


def test_generate_random():
    df = bigvul()
    df = bigvul_filter(
        df, check_file=True, check_valid=True, vulonly=False, load_code=False,
    )
    for split in ["random", "fixed"]:
        df = bigvul_partition(df, partition="all", split=split)
        print(split)
        print(df.value_counts("label", normalize=True))

def get_dataflow_output(_id):
    idpath = itempath(_id)
    dataflow_file = idpath.parent / (idpath.name + ".dataflow.json")
    with open(dataflow_file) as f:
        dataflow_data = json.load(f)
    updated_in = {}
    updated_out = {}
    for _, data in dataflow_data.items():
        data_out = data["solution.out"]
        assert len(set(updated_out.keys()) & set(data_out.keys())) == 0, "should be no overlap"
        updated_out.update(data_out)
        data_in = data["solution.in"]
        assert len(set(updated_in.keys()) & set(data_in.keys())) == 0, "should be no overlap"
        updated_in.update(data_in)
    updated_in = {int(k): v for k, v in updated_in.items()}
    updated_out = {int(k): v for k, v in updated_out.items()}
    return updated_in, updated_out

def test_debug():
    df = ds("mutated_var_rename")
    print(df)