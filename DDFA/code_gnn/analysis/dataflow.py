import networkx as nx

import sastvd.helpers.joern as svdj
import sastvd.helpers.datasets as svdds
import sastvd.helpers.dclass as svddc
import dataclasses


def get_edge_subgraph(cpg, graph_etype):
    filtered_edges = [
        (u, v, k)
        for u, v, k, etype in cpg.edges(keys=True, data="type")
        if etype == graph_etype
    ]
    return cpg.edge_subgraph(edges=filtered_edges)


"""
# Operators from https://github.com/joernio/joern/blob/master/joern-cli/src/main/resources/default.semantics
# "<operator>.sizeOf" reduces the FPs
# 1->-1 first parameter mapped to return value (-1)
# "<operator>.addition" 1->-1 2->-1
# "<operator>.addressOf" 1->-1
"<operator>.assignment" 2->1
"<operator>.assignmentAnd" 2->1 1->1
"<operator>.assignmentArithmeticShiftRight" 2->1 1->1
"<operator>.assignmentDivision" 2->1 1->1
"<operator>.assignmentExponentiation" 2->1 1->1
"<operator>.assignmentLogicalShiftRight" 2->1 1->1
"<operator>.assignmentMinus" 2->1 1->1
"<operator>.assignmentModulo" 2->1 1->1
"<operator>.assignmentMultiplication" 2->1 1->1
"<operator>.assignmentOr" 2->1 1->1
"<operator>.assignmentPlus" 2->1 1->1
"<operator>.assignmentShiftLeft" 2->1 1->1
"<operator>.assignmentXor" 2->1 1->1
# "<operator>.computedMemberAccess" 1->-1
# "<operator>.conditional" 2->-1 3->-1
# "<operator>.fieldAccess" 1->-1
# "<operator>.getElementPtr" 1->-1
"<operator>.incBy" 1->1 2->1 3->1 4->1
# "<operator>.indexAccess" 1->-1
# "<operator>.indirectComputedMemberAccess" 1->-1
# "<operator>.indirectFieldAccess" 1->-1
# "<operator>.indirectIndexAccess" 1->-1 2->-1
# "<operator>.indirectMemberAccess" 1->-1
# "<operator>.indirection" 1->-1
# "<operator>.memberAccess" 1->-1
# "<operator>.pointerShift" 1->-1
"<operator>.postDecrement" 1->1
"<operator>.postIncrement" 1->1
"<operator>.preDecrement" 1->1
"<operator>.preIncrement" 1->1
# "<operator>.sizeOf"
# "free" 1->1
# "scanf" 2->2
# "strcmp" 1->-1 2->-1
"""

assignment_ops = [
    "<operator>.assignment",
    "<operator>.assignmentAnd",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentDivision",
    "<operator>.assignmentExponentiation",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentMinus",
    "<operator>.assignmentModulo",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentOr",
    "<operator>.assignmentPlus",
    "<operator>.assignmentShiftLeft",
    "<operator>.assignmentXor",
]
inc_dec_ops = [
    "<operator>.incBy",
    "<operator>.postDecrement",
    "<operator>.postIncrement",
    "<operator>.preDecrement",
    "<operator>.preIncrement",
]
mod_ops = assignment_ops + inc_dec_ops
mod_ops += [op.replace("<operator>", "<operators>") for op in assignment_ops]
mod_ops += [op.replace("<operator>", "<operators>") for op in inc_dec_ops]


@dataclasses.dataclass
class VariableDefinition:
    v: str
    node: int
    code: str

    def __hash__(self):
        return self.node

    def __eq__(self, other):
        return self.node == other.node

    def __lt__(self, other):
        return self.node < other.node


class ReachingDefinitions:
    def __init__(self, cpg):
        self.cpg = cpg
        self.cfg = get_edge_subgraph(cpg, "CFG")
        self.ast = get_edge_subgraph(cpg, "AST")
        self.argument = get_edge_subgraph(cpg, "ARGUMENT")

        # Collect domain in constructor and index into it
        # instead of creating VariableDefinition during analysis
        self.gen_set = {}
        for node, attr in self.cpg.nodes(data=True):
            if attr["name"] in mod_ops:
                self.gen_set[node] = {
                    VariableDefinition(
                        self.get_assigned_variable(node),
                        node,
                        self.cpg.nodes[node]["code"],
                    )
                }
            else:
                self.gen_set[node] = set()

    @property
    def domain(self):
        return set().union(*self.gen_set.values())

    def get_assigned_variable(self, node):
        """Get the name of the variable assigned in the node, if any"""
        if node in self.ast.nodes:
            if self.cpg.nodes[node]["name"] in mod_ops:
                children = sorted(
                    self.argument.successors(node),
                    key=lambda n: self.cpg.nodes[n]["order"],
                )
                if len(children) > 0:
                    return self.ast.nodes[children[0]]["code"]
        return None

    def gen(self, node):
        """if v is defined in node, gen {node}"""
        return self.gen_set[node]

    def kill(self, node, definitions=None):
        """if v is defined in node, kill {all other definitions of v}"""
        if definitions is None:
            definitions = self.domain
        v = self.get_assigned_variable(node)
        if v is None:
            return set()
        else:
            return {d for d in definitions if d.v == v and d.node != node}

    def get_reaching_definitions(self):
        """https://www.cs.cmu.edu/afs/cs/academic/class/15745-s16/www/lectures/L6-Foundations-of-Dataflow.pdf"""
        out_reachingdefs = {}
        for n in self.cfg.nodes():
            out_reachingdefs[n] = set()

        in_reachingdefs = {}
        worklist = list(self.cfg.nodes())
        while len(worklist) > 0:
            n = worklist.pop()
            in_reachingdefs[n] = set()
            for p in self.cfg.predecessors(n):
                in_reachingdefs[n] = in_reachingdefs[n].union(out_reachingdefs[p])

            new_out_reaching_defs = self.gen(n).union(
                (in_reachingdefs[n].difference(self.kill(n, in_reachingdefs[n])))
            )
            if new_out_reaching_defs != out_reachingdefs[n]:
                for s in self.cfg.successors(n):
                    worklist.append(s)
            out_reachingdefs[n] = new_out_reaching_defs

        return in_reachingdefs

    def __str__(self):
        domain = self.domain
        return f"{len(domain)} defs: {str([d.code for d in domain])}"


def print_program(cpg):
    for p in sorted(cpg.nodes(data=True), key=lambda p: p[1].get("id", -1)):
        if "code" in p[1]:
            print(str(p[1]["lineNumber"]) + ": " + p[1]["code"])


def sub(cpg, etype):
    return nx.edge_subgraph(
        cpg,
        (
            (u, v, k)
            for u, v, k, attr in cpg.edges(keys=True, data=True)
            if attr["type"] == etype
        ),
    )


def get_cpg(_id, dsname="bigvul", return_n_e=False):
    n, e = svdj.get_node_edges(svdds.itempath(_id, dsname))

    # inline parts of this function to clean up nodes without grouping by lineno
    n = n[n.lineNumber != ""].copy()
    n.lineNumber = n.lineNumber.astype(int)
    e.innode = e.innode.astype(int)
    e.outnode = e.outnode.astype(int)
    n = svdj.drop_lone_nodes(n, e)
    e = e.drop_duplicates(subset=["innode", "outnode", "etype"])

    # e = svdj.rdg(e, "dataflow")
    n = svdj.drop_lone_nodes(n, e)

    # TODO: This is a stopgap. Find out why there are extra edges!
    e = e[e.innode.isin(n.id) & e.outnode.isin(n.id)]

    nodes = n
    edges = e

    # Extract CFG with code
    cpg = nx.MultiDiGraph()
    cpg.add_nodes_from(
        nodes.apply(
            lambda n: (
                n.id,
                {
                    "lineNumber": n.lineNumber
                    if isinstance(n.lineNumber, (int, float))
                    else None,
                    "code": n.code,
                    "name": n["name"],
                    "_label": n._label,
                    "order": int(n.order)
                    if isinstance(n.order, (int, float))
                    else None,
                    "typeFullName": n.typeFullName,
                },
            ),
            axis=1,
        )
    )
    cpg.add_edges_from(
        edges.apply(lambda e: (e.outnode, e.innode, {"type": e.etype}), axis=1)
    )

    if return_n_e:
        return cpg, n, e
    else:
        return cpg


def test_weird_assignment_operators():
    """
    For some reason the operators in this program show up as <operators> instead of <operator>.
    Make sure these are still detected.
    """
    cpg = get_cpg(svddc.svdds.itempath(18983))
    print(cpg)
    problem = ReachingDefinitions(cpg)
    print(problem)
    assert len(problem.domain) == 12


def test_get_cpg():
    cpg = get_cpg(svddc.svdds.itempath(0))
    print(cpg)
    problem = ReachingDefinitions(cpg)
    print(problem)

    gas = problem.get_assigned_variable(1000107)
    print("should get variable", gas)
    assert gas is not None

    gas2 = problem.get_assigned_variable(1000129)
    print("should not get variable", gas2)
    assert gas2 is None

    gen = problem.gen(1000107)
    print("should gen", gen)
    assert len(gen) == 1
    assert list(gen)[0].v == "schemaFlagsEx"

    gen = problem.gen(1000129)
    print("should not gen", gen)
    assert len(gen) == 0

    kill = problem.kill(1000107, problem.gen(1000107))
    print("should kill itself", kill)
    assert len(kill) == 1

    kill2 = problem.kill(
        1000107,
        problem.gen(1000107).union(
            {VariableDefinition("schemaFlagsEx", -1, "schemaFlagsEx = foo()")}
        ),
    )
    print("should kill itself and any others", kill2)
    assert len(kill2) == 2

    rd = problem.get_reaching_definitions()
    # print("should have reaching definitions", json.dumps({cpg.nodes[n]["lineNumber"]: [dataclasses.asdict(x) for x in sorted(d)] for n, d in rd.items()}, indent=2))
    assert len(rd) == len(problem.cfg.nodes)
    assert any(len(d) > 0 for d in rd.values())
    # This is only a simple test case which doesn't reassign any variables,
    # so we expect that every node has RD only from the nodes on previous lines.
    # Does not hold for all programs.
    nodes_and_counts = [
        (cpg.nodes[n], len(d))
        for n, d in rd.items()
        if cpg.nodes[n]["_label"] != "METHOD_RETURN"
    ]
    nodes_and_counts_by_lineno = sorted(
        nodes_and_counts, key=lambda p: p[0]["lineNumber"]
    )
    counts = [c for n, c in nodes_and_counts_by_lineno]
    assert counts == sorted(counts)
