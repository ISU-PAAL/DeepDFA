import networkx as nx


def normalize(cpg):
    cpg_nodes = cpg.nodes
    cpg_nodes_sorted = sorted(cpg_nodes)
    cpg_nodes_relabel = {n: cpg_nodes_sorted.index(n) for n in cpg.nodes}
    cpg = nx.relabel_nodes(cpg, cpg_nodes_relabel)
    # for n in cpg.nodes:
    #     cpg.nodes[n]["label"] = str(n)
    return cpg
