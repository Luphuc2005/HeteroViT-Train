import os
import unittest
from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.online_state import OnlineClusterState
from src.scheduler.online_cost_model import OnlineCostModel
from src.scheduler.dynamic_rebalancer import DynamicRebalancer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPUTE_PROFILE = os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json")
NETWORK_PROFILE = os.path.join(PROJECT_ROOT, "profiles", "network_profile.json")


class TestDynamicRebalancer(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(COMPUTE_PROFILE) or not os.path.exists(NETWORK_PROFILE):
            self.skipTest("Profile files missing")

        compute_model = ComputeCostModel(COMPUTE_PROFILE)
        comm_model = CommunicationCostModel(NETWORK_PROFILE)
        self.cluster_state = OnlineClusterState(ema_alpha=0.5)
        self.cost_model = OnlineCostModel(
            compute_model=compute_model,
            communication_model=comm_model,
            cluster_state=self.cluster_state,
            r_min=0.5,
            r_max=3.0,
            lambda_penalty=1.0,
        )

        self.initial_allocation = {
            "lab01": 384,
            "lab02": 64,
            "lab03": 48,
            "lab04": 48,
            "lab05": 96,
        }
        self.rebalancer = DynamicRebalancer(
            cost_model=self.cost_model,
            epsilon=0.05,
            cooldown_epochs=1,
        )

    def test_stable_execution_keeps_allocation(self):
        # Under normal conditions matching profile, scheduler should KEEP current allocation
        telemetry = [
            {"node_id": "lab01", "compute_ms": 658.0, "current_batch": 384, "rank": 0},
            {"node_id": "lab02", "compute_ms": 2385.0, "current_batch": 64, "rank": 1},
            {"node_id": "lab03", "compute_ms": 2126.0, "current_batch": 48, "rank": 2},
            {"node_id": "lab04", "compute_ms": 2119.0, "current_batch": 48, "rank": 3},
            {"node_id": "lab05", "compute_ms": 1482.0, "current_batch": 96, "rank": 4},
        ]
        decision = self.rebalancer.decide_rebalance(
            epoch=1,
            telemetry_list=telemetry,
            current_allocation=self.initial_allocation,
            global_batch=640,
        )
        self.assertIn(decision.action, ["KEEP", "SWITCH"])
        self.assertEqual(sum(decision.target_allocation.values()), 640)

    def test_slowdown_detection_and_switch(self):
        # Severe slowdown on lab03 (compute spikes 2.5x -> 5300ms)
        telemetry_slow = [
            {"node_id": "lab01", "compute_ms": 660.0, "current_batch": 384, "rank": 0},
            {"node_id": "lab02", "compute_ms": 2390.0, "current_batch": 64, "rank": 1},
            {"node_id": "lab03", "compute_ms": 5500.0, "current_batch": 48, "rank": 2},
            {"node_id": "lab04", "compute_ms": 2120.0, "current_batch": 48, "rank": 3},
            {"node_id": "lab05", "compute_ms": 1485.0, "current_batch": 96, "rank": 4},
        ]
        decision = self.rebalancer.decide_rebalance(
            epoch=2,
            telemetry_list=telemetry_slow,
            current_allocation=self.initial_allocation,
            global_batch=640,
        )
        self.assertEqual(decision.action, "SWITCH")
        self.assertGreater(decision.predicted_gain_pct, 5.0)
        self.assertLess(decision.target_allocation["lab03"], 48)
        self.assertEqual(sum(decision.target_allocation.values()), 640)

        # Immediate next epoch: Cooldown should prevent another switch
        decision_cd = self.rebalancer.decide_rebalance(
            epoch=3,
            telemetry_list=telemetry_slow,
            current_allocation=decision.target_allocation,
            global_batch=640,
        )
        self.assertEqual(decision_cd.action, "KEEP")
        self.assertIn("cooldown_active", decision_cd.reason)


if __name__ == "__main__":
    unittest.main()
