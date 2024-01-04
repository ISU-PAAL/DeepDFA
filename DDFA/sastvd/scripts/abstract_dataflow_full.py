"""
Extract abstract dataflow features from graphs
Yields:
- storage/cache/bigvul/abstract_dataflow.csv (cache)
- storage/processed/bigvul/abstract_dataflow_hash_api_datatype_literal_operator.csv
"""

import argparse
import functools
import json
import re
import traceback
from multiprocessing import Pool

import networkx as nx
import pandas as pd
import code_gnn.analysis.dataflow as dataflow
import sastvd.helpers.datasets as svdds
import sastvd as svd
import tqdm

# Extract dataflow features from CPG

all_assignment_types = (
    "<operator>.assignmentDivision",
    "<operator>.assignmentExponentiation",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentModulo",
    "<operator>.assignmentMultiplication",
    "<operator>.preIncrement",
    "<operator>.preDecrement",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.assignment",
    "<operator>.assignmentOr",
    "<operator>.assignmentAnd",
    "<operator>.assignmentXor",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentShiftLeft",
)

def is_decl(n_attr):
    # NOTE: this is local variable declarationsm
    # which are not considered definitions in formal DFA setting.
    # if n_attr["_label"] in ("LOCAL",):
    #     return True

    # https://github.com/joernio/joern/blob/15e241d3174ecba9e977a399793c9c6a1249d819/semanticcpg/src/main/scala/io/shiftleft/semanticcpg/language/operatorextension/package.scala
    return n_attr["_label"] == "CALL" and n_attr["name"] in all_assignment_types


def get_dataflow_features(graph_id, raise_all=False, verbose=False):
    try:
        # breakpoint()
        cpg, n, e = dataflow.get_cpg(graph_id, dsname, return_n_e=True)
        # print(cpg)
        # print(n)
        # print(e)
        ast = dataflow.sub(cpg, "AST")
        arg_graph = dataflow.sub(cpg, "ARGUMENT")
        labels = nx.get_node_attributes(cpg, "_label")
        code = nx.get_node_attributes(cpg, "code")
        names = nx.get_node_attributes(cpg, "name")

        def recurse_datatype(v):
            v_attr = cpg.nodes[v]
            if verbose:
                print("recursing", v, v_attr)

            name_idx = {
                "<operator>.indirectIndexAccess": 1,
                "<operator>.indirectFieldAccess": 1,
                "<operator>.indirection": 1,
                "<operator>.fieldAccess": 1,
                "<operator>.postIncrement": 1,
                "<operator>.postDecrement": 1,
                "<operator>.preIncrement": 1,
                "<operator>.preDecrement": 1,
                "<operator>.addressOf": 1,
                "<operator>.cast": 2,
                "<operator>.addition": 1,
            }
            if v_attr["_label"] == "IDENTIFIER":
                return v, v_attr["typeFullName"]
            elif v_attr["_label"] == "CALL":
                if v_attr["name"] in name_idx.keys():
                    # TODO: Get field data type, not struct data type
                    args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(v)}
                    arg = args[name_idx[v_attr["name"]]]
                    arg_attr = cpg.nodes[arg]
                    if verbose:
                        print("index", arg, arg_attr)
                        if v_attr["name"] == "<operator>.addition":
                            print("addition debug", v, v_attr, arg, arg_attr)
                    if arg_attr["_label"] == "IDENTIFIER":
                        return arg, arg_attr["typeFullName"]
                    elif arg_attr["_label"] == "CALL":
                        return recurse_datatype(arg)
                    else:
                        raise NotImplementedError(
                            f"recurse_datatype index could not handle {v} {v_attr} -> {arg} {arg_attr}"
                        )
            raise NotImplementedError(
                f"recurse_datatype var could not handle {v} {v_attr}"
            )

        def get_raw_datatype(decl):
            decl_attr = cpg.nodes[decl]

            if verbose:
                print("parent", decl, decl_attr)

            if decl_attr["_label"] == "LOCAL":
                return decl, decl_attr["typeFullName"]
            elif decl_attr["_label"] == "CALL" and decl_attr[
                "name"
            ] in all_assignment_types + ("<operator>.cast",):
                args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(decl)}
                return recurse_datatype(args[1])
            else:
                raise NotImplementedError(
                    f"""get_raw_datatype did not handle {decl} {decl_attr}"""
                )

        def grab_declfeats(node_id):
            fields = []
            try:
                ret = get_raw_datatype(node_id)
                if ret is not None:
                    child_id, child_datatype = ret
                    fields.append(("datatype", child_id, child_datatype))

                # create a copy of the AST with method definitions excluded.
                # this avoids an issue where some variable definitions descend to
                # method definitions (probably by mistake), shown in graph 3.
                my_ast = ast.copy()
                my_ast.remove_nodes_from(
                    [
                        n
                        for n, attr in ast.nodes(data=True)
                        if attr["_label"] == "METHOD"
                    ]
                )

                to_search = nx.descendants(my_ast, node_id)
                for n in to_search:
                    if verbose:
                        print(
                            f"{node_id} desc {n} {code.get(n, None)} {names.get(n, None)} {nx.shortest_path(ast, node_id, n)}"
                        )
                    if labels[n] == "LITERAL":
                        fields.append(("literal", n, code.get(n, pd.NA)))
                    if labels[n] == "CALL":
                        if m := re.match(r"<operator>\.(.*)", names[n]):
                            operator_name = m.group(1)
                            if operator_name not in ("indirection",):
                                fields.append(("operator", n, operator_name))
                        # handle API call
                        else:
                            fields.append(("api", n, names[n]))
            except Exception:
                print("node error", node_id, traceback.format_exc())
                if raise_all:
                    raise
            return fields

        # nx.set_node_attributes(
        #     ast,
        #     {n: f"{n}: {attr['code']}" for n, attr in ast.nodes(data=True)},
        #     "label",
        # )
        # A = nx.drawing.nx_agraph.to_agraph(ast)
        # A.layout("dot")
        # A.draw("abcd.png")

        # n = n.rename(columns={"id": "node_id"})
        n["graph_id"] = graph_id
        # print("select nodes")
        # print(n["node_id"].isin(n for n, attr in cpg.nodes(data=True) if is_decl(attr)))
        decls = n[
            n["node_id"].isin(n for n, attr in cpg.nodes(data=True) if is_decl(attr))
        ].copy()
        decls["fields"] = decls["node_id"].apply(grab_declfeats)
        decls = decls.explode("fields").dropna()
        if verbose: print("extracted fields:", decls["fields"], sep="\n")
        if len(decls) > 0:
            decls["subkey"], decls["subkey_node_id"], decls["subkey_text"] = zip(
                *decls["fields"]
            )
        else:
            decls["subkey"] = None
            decls["subkey_node_id"] = None
            decls["subkey_text"] = None
        return decls
    except Exception:
        print("graph error", graph_id, traceback.format_exc())
        if raise_all:
            raise


# Get all abstract dataflow info
def get_dataflow_features_df():
    # print(get_dataflow_features_df)
    csv_file = (
        svd.cache_dir() / f"{dsname}/abstract_dataflow{'_sample' if args.sample else ''}.csv"
    )
    if csv_file.exists() and args.cache:
        dataflow_df = pd.read_csv(csv_file)
    else:
        dataflow_df = pd.DataFrame()
        all_df = svdds.ds(dsname, sample=args.sample)
        with Pool(args.workers) as pool:
            for decls_df in tqdm.tqdm(
                pool.imap(
                # map(
                    functools.partial(
                        get_dataflow_features,
                        raise_all=args.sample,
                        verbose=args.verbose,
                    ),
                    all_df.id,
                ),
                total=len(all_df),
                desc="get abstract dataflow features",
            ):
                dataflow_df = pd.concat([dataflow_df, decls_df], ignore_index=True)

        dataflow_df = dataflow_df[
            ["graph_id", "node_id", "subkey", "subkey_node_id", "subkey_text"]
        ]

        csv_file.parent.mkdir(exist_ok=True, parents=True)
        dataflow_df.to_csv(csv_file)

    return dataflow_df


def cleanup_datatype(df):
    """Assign datatype to cleaned-up version"""
    df.loc[df["subkey"] == "datatype", "subkey_text"] = dataflow_df[
        "subkey_text"
    ].apply(
        lambda dt: dt
        if pd.isna(dt)
        else re.sub(
            r"\s+", r" ", re.sub(r"^const ", r"", re.sub(r"\s*\[.*\]", r"[]", dt))
        ).strip()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstract dataflow")
    parser.add_argument("--dsname", default="bigvul")
    parser.add_argument("--sample", action="store_true", help="Extract sample only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.set_defaults(cache=True)
    parser.add_argument(
        "--workers", type=int, default=6, help="How many workers to use"
    )
    parser.add_argument("--stage", type=int, default=1, help="Which stages to execute")
    parser.add_argument("--select_subkeys", nargs="+", default=["api", "datatype", "literal", "operator"], help="Which subkeys to export")
    args = parser.parse_args()

    dsname = args.dsname

    args.select_subkeys = sorted(args.select_subkeys)

    dataflow_df = get_dataflow_features_df()
    print("dataflow_df", dataflow_df)
    print("dataflow_df counts", dataflow_df.value_counts("subkey"))
    print("dataflow_df na", dataflow_df[dataflow_df["subkey_text"].isna()])

    if args.stage <= 1:
        exit()

"""
generate hash value for each node
"""


def to_hash(group, select_subkeys):
    # print(group)
    _hash = {
        subkey: sorted(
            [
                s for s in group[group["subkey"] == subkey]["subkey_text"].tolist()
            ]
        )
        for subkey in select_subkeys
    }
    return json.dumps(_hash)


if __name__ == "__main__":
    # get most common subkeys
    # TODO: don't filter out missing files/graphs
    # source = svddc.BigVulDataset(partition="sample" if args.sample else "train", undersample=False)
    # source_df = source.df
    # print("generate hash from train", source_df, sep="\n")

    # source_df = pd.merge(source_df, dataflow_df, left_on="id", right_on="graph_id")
    # select_key = "datatype"
    # source_vc = source_df[source_df["subkey"] == select_key].value_counts("subkey_text")
    # print("train values", source_vc)


    # Export dataset
    # TODO: export more combinations of subkeys
    # select_subkeys = ["datatype", "operator", "api", "literal"]
    # select = {
    #     select_key: source_vc.index.sort_values().tolist(),
    # }
    hashes = dataflow_df.groupby(["graph_id", "node_id"]).apply(to_hash, select_subkeys=args.select_subkeys)
    all_df = dataflow_df.set_index(["graph_id", "node_id"]).join(
        hashes.to_frame("hash")
    ).reset_index()
    print("Got hashes")
    print(all_df)
    all_df = (
        all_df[["graph_id", "node_id", "hash"]]
        .sort_values(by=["graph_id", "node_id"])
        .reset_index(drop=True)
    )[["graph_id", "node_id", "hash"]].drop_duplicates()
    print("hash result")
    print(all_df)
    print(all_df["hash"].value_counts(dropna=False, normalize=True))

    all_df.to_csv(
        svd.get_dir(svd.processed_dir() / dsname) / f"abstract_dataflow_hash_{'_'.join(args.select_subkeys)}{'_sample' if args.sample else ''}.csv"
    )
