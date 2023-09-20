import json
import random
import shutil

# from graphviz import Digraph
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sastvd as svd
import scipy.sparse as sparse
import pexpect


def nodelabel2line(label: str):
    """Given a node label, return the line number.

    Example:
    s = "METHOD_1.0: static long main()..."
    nodelabel2line(s)
    >>> '1.0'
    """
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]


def randcolor():
    """Generate random color."""

    def r():
        return random.randint(0, 255)

    return "#%02X%02X%02X" % (r(), r(), r())


def get_digraph(nodes, edges, edge_label=True):
    """Plote digraph given nodes and edges list."""
    dot = Digraph(comment="Combined PDG")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = randcolor()

    for n in nodes:
        style = {"style": "filled", "fillcolor": colormap[n[2]]}
        dot.node(str(n[0]), str(n[1]), **style)
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "CFG":
            style["style"] = "solid"
            style["color"] = "red"
        elif e[2] == "CDG":
            style["style"] = "solid"
            style["color"] = "blue"
        elif e[2] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    return dot


def run_joern_gettype(sess, cpgpath: str, datatypes, cache):
    """Extract member types using Joern."""
    datatype_to_paths = {}
    try:
        sess.import_cpg(cpgpath)
        try:
            for datatype in datatypes:
                typepath = None
                try:
                    typepath = Path(f"{cpgpath}_types_{datatype}.txt")
                    if not cache or not typepath.exists():
                        sess.run_script(
                            "get_type",
                            params={"rootType": datatype, "outFile": str(typepath)},
                            import_first=False,
                        )
                    datatype_to_paths[datatype] = typepath
                except pexpect.exceptions.EOF:
                    raise
                except Exception:
                    print("error", datatype, traceback.format_exc())
        finally:
            sess.delete()

        print("output to", typepath)

        def get_subtypes(typepath):
            if typepath is None:
                return None
            with open(typepath) as f:
                return list(f.read().splitlines())

    except pexpect.exceptions.EOF:
        raise
    except Exception:
        print("error", cpgpath, datatypes, traceback.format_exc())

    datatype_to_subtypes = {}
    for k, v in datatype_to_paths.items():
        try:
            subtypes = get_subtypes(v)
            if subtypes is not None:
                datatype_to_subtypes[k] = subtypes
        except Exception:
            print("error", k, v, traceback.format_exc())
    return datatype_to_subtypes


def run_joern_dataflow(sess, filepath: str, problem: str, verbose: int):
    """Extract graph using most recent Joern."""
    try:
        sess.import_code(filepath)
        sess.run_script(
            "get_dataflow_output", params={"filename": filepath, "problem": problem}
        )
    finally:
        sess.delete()


def run_joern_sess(
    sess, filepath: str, verbose: int, export_json: bool, export_cpg: bool, export_dataflow: bool
):
    """Extract graph using most recent Joern."""
    output = sess.run_script(
        "get_func_graph",
        params={
            "filename": filepath,
            "exportJson": export_json,
            "exportCpg": export_cpg,
            "exportDataflow": export_dataflow,
            "overwrite": True,
        },
        import_first=False,
    )
    if verbose >= 4:
        print(output)


def run_joern(filepath: str, verbose: int):
    """Extract graph using most recent Joern."""
    script_file = (
        (svd.external_dir() / "get_func_graph.scala").resolve().relative_to(Path.cwd())
    )
    filename = svd.external_dir() / filepath
    params = f"filename={filename}"
    command = f"joern --script {script_file} --params='{params}'"
    command = str(svd.external_dir() / "joern-cli" / command)
    if verbose > 2:
        svd.debug(command)
    svd.subprocess_cmd(command, verbose=verbose)
    try:
        shutil.rmtree(svd.external_dir() / "joern-cli" / "workspace" / filename.name)
    except Exception as E:
        if verbose > 4:
            print(E)
        pass


def get_node_edges(filepath: str, verbose=0):
    """Get node and edges given filepath (must run after run_joern).

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/53.c"
    """
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name

    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
        edges = edges.fillna("")

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        keepcols = [
            "id",
            "_label",
            "name",
            "code",
            "lineNumber",
            "columnNumber",
            "lineNumberEnd",
            "columnNumberEnd",
            "controlStructureType",
            "order",
            "fullName",
            "typeFullName",
        ]
        nodes = pd.DataFrame.from_records(nodes, columns=keepcols)
        # if "controlStructureType" not in nodes.columns:
        #    nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")
        try:
            # actual_keepcols = []
            # for col in keepcols:
            #    if col in nodes.columns:
            #        actual_keepcols.append(col)
            #    else:
            #        print("warn:", col, "not in columns", nodes.columns)
            # keepcols = actual_keepcols
            nodes = nodes[keepcols]
        except Exception as E:
            if verbose > 1:
                svd.debug(f"Failed {filepath}: {E} {traceback.format_exc()}")
            return None

    if len(nodes[nodes._label == "METHOD"]) == 0:
        raise Exception("empty graph")

    # Assign line number to local variables
    with open(filepath, "r") as f:
        code = f.readlines()
    lmap = assign_line_num_to_local(nodes, edges, code, filepath=filepath)
    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )
    nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            nodes = pd.concat(
                (
                    nodes,
                    pd.DataFrame(
                        {
                            "id": e.outnode,
                            "node_label": node_label,
                            "lineNumber": lineNum,
                        },
                        index=[0],
                    ),
                ),
                ignore_index=True,
            )

    # moved some stuff from ne_groupnodes to here
    nodes[
        "node_id"
    ] = nodes.id  # id column is later overwritten to lineNumer, so saving it here
    nodes.lineNumber = (
        pd.to_numeric(nodes.lineNumber, errors="coerce").fillna(-1).astype(int)
    )
    nodes = nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nodes = drop_lone_nodes(nodes, edges)
    edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"])

    # breakpoint()
    # TODO: test if this is necessary... check in_line and innode to see which one is float
    # if "innode" in edges.columns:
    #     edges = edges[edges.innode.apply(lambda x: isinstance(x, float))]
    #     edges.innode = edges.innode.astype(int)
    # if "outnode" in edges.columns:
    #     edges = edges[edges.outnode.apply(lambda x: isinstance(x, float))]
    #     edges.outnode = edges.outnode.astype(int)

    return nodes, edges


def plot_node_edges(filepath: str, lineNumber: int = -1, filter_edges=[]):
    """Plot node edges given filepath (must run after get_node_edges).

    TO BE DEPRECATED.
    """
    nodes, edges = get_node_edges(filepath)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
    ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    dot.render("/tmp/tmp.gv", view=True)


def full_run_joern(filepath: str, verbose=0):
    """Run full Joern extraction and save output."""
    try:
        run_joern(filepath, verbose)
        nodes, edges = get_node_edges(filepath, verbose=verbose)
        return {"nodes": nodes, "edges": edges}
    except Exception as E:
        if verbose > 0:
            svd.debug(f"Failed {filepath}: {E} {traceback.format_exc()}")
        return None


def full_run_joern_from_string(code: str, dataset: str, iid: str, verbose=0):
    """Run full joern from a string instead of file."""
    savedir = svd.get_dir(svd.interim_dir() / dataset)
    savepath = savedir / f"{iid}.c"
    with open(savepath, "w") as f:
        f.write(code)
    return full_run_joern(savepath, verbose)


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours.

    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours


def rdg(edges, gtype):
    """Reduce graph given type."""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "cfg":
        return edges[(edges.etype == "CFG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
        ]
    if gtype == "dataflow":
        # return edges[(edges.etype == "CFG") | (edges.etype == "AST") | (edges.etype == "ARGUMENT")]
        return edges[(edges.etype == "CFG") | (edges.etype == "AST")]


def assign_line_num_to_local(nodes, edges, code, filepath=None):
    """Assign line number to local variable in CPG."""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
    ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption {}.".format(filepath)
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption {}.".format(filepath)
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map


def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def plot_graph_node_edge_df(
    nodes, edges, nodeids=[], hop=1, drop_lone_nodes=True, edge_label=True
):
    """Plot graph from node and edge dataframes.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
        drop_lone_nodes (bool): hide nodes with no in/out edges.
        lineNumber (int): Plot subgraph around this node.
    """
    # Drop lone nodes
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Get subgraph
    if len(nodeids) > 0:
        nodeids = nodes[nodes.lineNumber.isin(nodeids)].id
        keep_nodes = neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
        ]

    dot = get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label,
    )
    dot.render("/tmp/tmp.gv", view=True)
