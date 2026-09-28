import unittest
from src.scheduler.online_state import NodeRuntimeState


class TestOnlineState(unittest.TestCase):
    def test_initialization(self):
        state = NodeRuntimeState(rank=0, node_id="lab01", current_batch=256, alpha=0.2)
        self.assertEqual(state.rank, 0)
        self.assertEqual(state.node_id, "lab01")
        self.assertEqual(state.current_batch, 256)
        self.assertTrue(state.active)
        self.assertEqual(state.compute_ema_ms, 0.0)

    def test_ema_update(self):
        state = NodeRuntimeState(rank=1, node_id="lab02", current_batch=32, alpha=0.5)
        # First observation: EMA should equal observed value
        state.update(observed_compute_ms=100.0, observed_comm_ms=10.0, observed_idle_ms=20.0)
        self.assertEqual(state.compute_ms, 100.0)
        self.assertEqual(state.compute_ema_ms, 100.0)
        self.assertEqual(state.sample_count, 1)

        # Second observation: 0.5 * 200.0 + 0.5 * 100.0 = 150.0
        state.update(observed_compute_ms=200.0, observed_comm_ms=12.0, observed_idle_ms=5.0)
        self.assertEqual(state.compute_ms, 200.0)
        self.assertEqual(state.compute_ema_ms, 150.0)
        self.assertEqual(state.sample_count, 2)

    def test_inactive_node(self):
        state = NodeRuntimeState(rank=2, node_id="lab03", current_batch=0)
        self.assertFalse(state.active)
        state.update(observed_compute_ms=0.0, current_batch=0)
        self.assertFalse(state.active)

        state.update(observed_compute_ms=50.0, current_batch=16)
        self.assertTrue(state.active)
        self.assertEqual(state.current_batch, 16)


if __name__ == "__main__":
    unittest.main()
