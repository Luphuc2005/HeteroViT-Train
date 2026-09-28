import os
import unittest
from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.online_state import OnlineClusterState
from src.scheduler.online_cost_model import OnlineCostModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPUTE_PROFILE = os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json")
NETWORK_PROFILE = os.path.join(PROJECT_ROOT, "profiles", "network_profile.json")


class TestOnlineCostModel(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(COMPUTE_PROFILE) or not os.path.exists(NETWORK_PROFILE):
            self.skipTest("Profile files missing")

        self.compute_model = ComputeCostModel(COMPUTE_PROFILE)
        self.comm_model = CommunicationCostModel(NETWORK_PROFILE)
        self.cluster_state = OnlineClusterState(ema_alpha=0.5)
        self.cost_model = OnlineCostModel(
            compute_model=self.compute_model,
            communication_model=self.comm_model,
            cluster_state=self.cluster_state,
            r_min=0.5,
            r_max=3.0,
            lambda_penalty=1.0,
        )

    def test_correction_factor_clamping(self):
        # lab01: profile compute at 256 is ~399ms. If EMA is 800ms -> raw_r ~ 2.0 (inside range)
        self.cluster_state.register_node(0, "lab01", 256)
        self.cluster_state.update_node("lab01", compute_ms=800.0, comm_ms=0, idle_ms=0, current_batch=256)

        # lab02: profile compute at 32 is ~1202ms. If EMA is 5000ms -> raw_r ~ 4.15 -> clamped to 3.0
        self.cluster_state.register_node(1, "lab02", 32)
        self.cluster_state.update_node("lab02", compute_ms=5000.0, comm_ms=0, idle_ms=0, current_batch=32)

        # lab03: profile compute at 24 is ~1053ms. If EMA is 200ms -> raw_r ~ 0.19 -> clamped to 0.5
        self.cluster_state.register_node(2, "lab03", 24)
        self.cluster_state.update_node("lab03", compute_ms=200.0, comm_ms=0, idle_ms=0, current_batch=24)

        r_lab01 = self.cost_model.compute_correction_factor("lab01")
        r_lab02 = self.cost_model.compute_correction_factor("lab02")
        r_lab03 = self.cost_model.compute_correction_factor("lab03")

        self.assertAlmostEqual(r_lab01, 800.0 / self.compute_model.predict_compute_ms("lab01", 256), delta=0.01)
        self.assertEqual(r_lab02, 3.0)
        self.assertEqual(r_lab03, 0.5)

    def test_evaluate_and_drop_worker(self):
        # 5-node candidate
        c_5node = {"lab01": 256, "lab02": 16, "lab03": 24, "lab04": 24, "lab05": 48}
        ev5 = self.cost_model.evaluate_candidate(c_5node)
        self.assertEqual(len(ev5.active_nodes), 5)
        self.assertEqual(ev5.total_batch, 368)
        self.assertGreater(ev5.t_critical_ms, 0)

        # 4-node candidate (lab02 dropped: batch 0)
        c_4node = {"lab01": 256, "lab02": 0, "lab03": 32, "lab04": 32, "lab05": 64}
        ev4 = self.cost_model.evaluate_candidate(c_4node)
        self.assertEqual(len(ev4.active_nodes), 4)
        self.assertIn("lab02", ev4.inactive_nodes)
        self.assertEqual(ev4.node_breakdown["lab02"]["batch"], 0)
        self.assertEqual(ev4.node_breakdown["lab02"]["compute_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
