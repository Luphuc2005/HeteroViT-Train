import os
import unittest
from src.scheduler.graph.cluster_graph import ClusterGraph, NodeVertex, EdgeLink

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ClusterGraphTests(unittest.TestCase):
    def setUp(self):
        self.profile_path = os.path.join(PROJECT_ROOT, "profiles", "cluster_graph.json")

    def test_load_cluster_graph(self):
        if not os.path.exists(self.profile_path):
            self.skipTest("cluster_graph.json not found")

        graph = ClusterGraph.load_from_json(self.profile_path)
        self.assertEqual(len(graph.nodes), 5)
        self.assertIn("lab01", graph.nodes)
        self.assertIn("lab02", graph.nodes)
        self.assertIn("lab03", graph.nodes)
        self.assertIn("lab04", graph.nodes)
        self.assertIn("lab05", graph.nodes)

        # Check nodes
        lab01 = graph.get_node("lab01")
        self.assertIsNotNone(lab01)
        self.assertEqual(lab01.rank, 0)
        self.assertEqual(lab01.device_type, "2xGPU")
        self.assertIn(256, lab01.available_batches)

        # Check edges
        edge = graph.get_edge("lab01", "lab02")
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge.bandwidth_10mb_mb_s, 11.201, places=2)

        # Bidirectional check
        edge_rev = graph.get_edge("lab02", "lab01")
        self.assertIsNotNone(edge_rev)
        self.assertEqual(edge.edge_id, edge_rev.edge_id)

    def test_update_runtime(self):
        if not os.path.exists(self.profile_path):
            self.skipTest("cluster_graph.json not found")

        graph = ClusterGraph.load_from_json(self.profile_path)
        graph.update_node_runtime("lab03", runtime_ema_ms=1250.0, runtime_std_ms=15.0, active=True)
        lab03 = graph.get_node("lab03")
        self.assertEqual(lab03.runtime_ema_ms, 1250.0)
        self.assertEqual(lab03.runtime_std_ms, 15.0)

    def test_serialization(self):
        if not os.path.exists(self.profile_path):
            self.skipTest("cluster_graph.json not found")

        graph = ClusterGraph.load_from_json(self.profile_path)
        d = graph.to_dict()
        self.assertEqual(len(d["nodes"]), 5)
        self.assertEqual(len(d["edges"]), 10)


if __name__ == "__main__":
    unittest.main()
