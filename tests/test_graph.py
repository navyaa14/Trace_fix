import tests._pathfix
import unittest

from graph import WorkflowGraph, NodeSpec, EdgeType, build_support_pipeline


class TestWorkflowGraph(unittest.TestCase):
    def _linear_graph(self):
        g = WorkflowGraph()
        g.add_node(NodeSpec("a", "kind"))
        g.add_node(NodeSpec("b", "kind"))
        g.add_node(NodeSpec("c", "kind"))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        return g

    def test_add_edge_requires_both_endpoints_to_exist(self):
        g = WorkflowGraph()
        g.add_node(NodeSpec("a", "kind"))
        with self.assertRaises(AssertionError):
            g.add_edge("a", "nonexistent")

    def test_topological_order_respects_all_edges(self):
        g = self._linear_graph()
        order = g.topological_order()
        self.assertEqual(order.index("a") < order.index("b") < order.index("c"), True)

    def test_topological_order_includes_every_node_exactly_once(self):
        g = self._linear_graph()
        order = g.topological_order()
        self.assertEqual(sorted(order), ["a", "b", "c"])
        self.assertEqual(len(order), len(set(order)))

    def test_topological_order_on_diamond_shape(self):
        g = WorkflowGraph()
        for n in "abcd":
            g.add_node(NodeSpec(n, "kind"))
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "d")
        g.add_edge("c", "d")
        order = g.topological_order()
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("a"), order.index("c"))
        self.assertLess(order.index("b"), order.index("d"))
        self.assertLess(order.index("c"), order.index("d"))

    def test_topological_order_silently_drops_nodes_in_a_cycle(self):
        g = WorkflowGraph()
        for n in "xyz":
            g.add_node(NodeSpec(n, "kind"))
        g.add_edge("x", "y")
        g.add_edge("y", "z")
        g.add_edge("z", "x")
        order = g.topological_order()
        self.assertLess(len(order), 3)

    def test_successors_and_predecessors(self):
        g = self._linear_graph()
        self.assertEqual(g.successors("a"), ["b"])
        self.assertEqual(g.predecessors("b"), ["a"])
        self.assertEqual(g.successors("c"), [])
        self.assertEqual(g.predecessors("a"), [])

    def test_build_support_pipeline_every_edge_endpoint_exists(self):
        g = build_support_pipeline()
        for e in g.edges:
            self.assertIn(e.src, g.nodes, f"edge source {e.src!r} not a node")
            self.assertIn(e.dst, g.nodes, f"edge dest {e.dst!r} not a node")

    def test_build_support_pipeline_is_acyclic(self):
        g = build_support_pipeline()
        order = g.topological_order()
        self.assertEqual(len(order), len(g.nodes),
                          "toposort dropped nodes -- graph has a cycle")

    def test_build_support_pipeline_human_is_a_sink(self):
        g = build_support_pipeline()
        self.assertEqual(g.successors("human"), [])

    def test_build_support_pipeline_has_expected_edge_type_on_escalation(self):
        g = build_support_pipeline()
        escalation = [e for e in g.edges if e.dst == "human"]
        self.assertEqual(len(escalation), 1)
        self.assertEqual(escalation[0].edge_type, EdgeType.ESCALATES_TO)


if __name__ == "__main__":
    unittest.main()
