import os
import pickle as pkl

import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.tokenise as svdt


import pickle as pkl
from pathlib import Path

import networkx as nx
import pandas as pd
import sastvd as svd
import sastvd.helpers.joern as svdj



def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation.

    DEBUGGING:
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/182480.c"

    PRINTING:
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "ast"), [24], 0)
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "reftype"))
    pd.options.display.max_colwidth = 500
    print(subseq.to_markdown(mode="github", index=0))
    print(nametypes.to_markdown(mode="github", index=0))
    print(uedge.to_markdown(mode="github", index=0))

    4/5 COMPARISON:
    Theirs: 31, 22, 13, 10, 6, 29, 25, 23
    Ours  : 40, 30, 19, 14, 7, 38, 33, 31
    Pred  : 40,   , 19, 14, 7, 38, 33, 31
    """
    cache_name = "_".join(str(filepath).split("/")[-3:])
    cachefp = svd.get_dir(svd.cache_dir() / "ivdetect_feat_ext") / Path(cache_name).stem
    try:
        with open(cachefp, "rb") as f:
            return pkl.load(f)
    except:
        pass

    try:
        nodes, edges = svdj.get_node_edges(filepath)
    except:
        return None

    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "code", "local_type"]].copy()
    subseq.code = subseq.local_type + " " + subseq.code
    subseq = subseq.drop(columns="local_type")
    subseq = subseq[~subseq.eq("").any(1)]
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    subseq = subseq.sort_values("lineNumber")
    subseq.code = subseq.code.apply(svdt.tokenise)
    subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. Line to AST
    ast_edges = svdj.rdg(edges, "ast")
    ast_nodes = svdj.drop_lone_nodes(nodes, ast_edges)
    ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]
    ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()
    ast_edges.innode = ast_edges.innode.map(ast_dict)
    ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})
    ast_nodes.code = ast_nodes.code.fillna("").apply(svdt.tokenise)
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )
    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})
    ast = ast_edges.join(ast_nodes, how="inner")
    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    ast = ast.to_dict()["ast"]

    # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # incoming connections (parent node), then add an edge from that node to the node
    # with id = 0 (unless it is zero itself).
    # DEBUG:
    # import sastvd.helpers.graphs as svdgr
    # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    for k, v in ast.items():
        allnodes = nodes_per_line[k]
        outnodes = v[0]
        innodes = v[1]
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        parentnodes = [i for i in outnodes if i not in innodes]
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    reftype_edges = svdj.rdg(edges, "reftype")
    reftype_nodes = svdj.drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))
    varnametypes = list()
    for cc in reftype_cc:
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].name.item()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    nametypes.type = nametypes.type.apply(svdt.tokenise)
    nametypes.name = nametypes.name.apply(svdt.tokenise)
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    nametypes = nametypes.to_dict()["nametype"]

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = svdj.rdg(edgesline, "pdg")
    nodesline = svdj.drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edgesline["etype"] = edgesline.apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
    edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # Cache
    with open(cachefp, "wb") as f:
        pkl.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges


def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """Get lines that are dependent on added lines.

    Example:
    df = svdd.bigvul()
    filepath_before = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/177775.c"
    filepath_after = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    """
    before_graph = feature_extraction(filepath_before)[0]
    after_graph = feature_extraction(filepath_after)[0]

    # Get nodes in graph corresponding to added lines
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # Get lines dependent on added lines in added graph
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # Filter by lines in before graph
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def helper(row):
    """Run get_dep_add_lines from dict.

    Example:
    df = svdd.bigvul()
    added = df[df.id==177775].added.item()
    removed = df[df.id==177775].removed.item()
    helper({"id":177775, "removed": removed, "added": added})
    """
    before_path = str(svd.processed_dir() / f"bigvul/before/{row['id']}.c")
    after_path = str(svd.processed_dir() / f"bigvul/after/{row['id']}.c")
    try:
        dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
    except Exception:
        dep_add_lines = []
    return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]


def get_dep_add_lines_bigvul(dataset, cache=True, sample=False):
    """Cache dependent added lines for a dataset."""
    saved = (
        svd.get_dir(svd.processed_dir()/dataset/"eval")
        / f"statement_labels{'_sample' if sample else ''}.pkl"
    )
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = svdd.bigvul(sample=sample)
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = svd.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
    return lines_dict


# def get_dep_add_lines_bigvul(cache=True, sample=False):
#     return get_dep_add_lines("bigvul", cache, sample)


def eval_statements(sm_logits, labels, thresh=0.5):
    """Evaluate statement-level detection according to IVDetect.

    sm_logits = [
        [0.5747372, 0.4252628],
        [0.53908646, 0.4609135],
        [0.49043426, 0.5095658],
        [0.65794635, 0.34205365],
        [0.3370166, 0.66298336],
        [0.55573744, 0.4442625],
    ]
    labels = [0, 0, 0, 0, 1, 0]
    """
    if sum(labels) == 0:
        preds = [i for i in sm_logits if i[1] > thresh]
        if len(preds) > 0:
            ret = {k: 0 for k in range(1, 11)}
        else:
            ret = {k: 1 for k in range(1, 11)}
    else:
        zipped = list(zip(sm_logits, labels))
        zipped = sorted(zipped, key=lambda x: x[0][1], reverse=True)
        ret = {}
        for i in range(1, 11):
            if 1 in [i[1] for i in zipped[:i]]:
                ret[i] = 1
            else:
                ret[i] = 0
    return ret


def eval_statements_inter(stmt_pred_list, thresh=0.5):
    """Intermediate calculation."""
    total = len(stmt_pred_list)
    ret = {k: 0 for k in range(1, 11)}
    for item in stmt_pred_list:
        eval_stmt = eval_statements(item[0], item[1], thresh)
        for i in range(1, 11):
            ret[i] += eval_stmt[i]
    ret = {k: v / total for k, v in ret.items()}
    return ret


def eval_statements_list(stmt_pred_list, thresh=0.5, vo=False):
    """Apply eval statements to whole list of preds.

    item1 = [[[0.1, 0.9], [0.6, 0.4], [0.4, 0.5]], [0, 1, 1]]
    item2 = [[[0.9, 0.1], [0.6, 0.4]], [0, 0]]
    item3 = [[[0.1, 0.9], [0.6, 0.4]], [1, 1]]
    stmt_pred_list = [item1, item2, item3]
    """
    vo_list = [i for i in stmt_pred_list if sum(i[1]) > 0]
    vulonly = eval_statements_inter(vo_list, thresh)
    if vo:
        return vulonly
    nvo_list = [i for i in stmt_pred_list if sum(i[1]) == 0]
    nonvulnonly = eval_statements_inter(nvo_list, thresh)
    ret = {}
    for i in range(1, 11):
        ret[i] = vulonly[i] * nonvulnonly[i]
    return ret
