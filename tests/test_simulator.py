from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.simulator import (  # noqa: E402
    Policy,
    Request,
    SimulationConfig,
    StageConfig,
    choose_backlog_aware_batch,
    run_simulation,
)


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            retrieval=StageConfig((1, 2, 4, 8), 4, 0.1, 0.3, partition_load_seconds=0.1),
            generation=StageConfig((1, 2, 4, 8), 4, 0.5, 0.9),
        )
        self.requests = tuple(Request(index, index * 0.01) for index in range(12))

    def test_all_policies_complete_each_request_once(self) -> None:
        for policy in Policy:
            result = run_simulation(self.requests, self.config, policy)
            self.assertEqual([item.request_id for item in result.timings], list(range(12)))
            self.assertTrue(all(item.latency_seconds >= 0 for item in result.timings))

    def test_adaptive_batch_is_feasible(self) -> None:
        choice = choose_backlog_aware_batch(5, self.config.generation)
        self.assertIn(choice, (1, 2, 4))
        self.assertLessEqual(choice, 5)

    def test_pipeline_reduces_latency_for_backlogged_workload(self) -> None:
        serial = run_simulation(self.requests, self.config, Policy.SERIAL)
        pipelined = run_simulation(self.requests, self.config, Policy.PIPELINED_STATIC)
        self.assertLess(pipelined.average_latency_seconds, serial.average_latency_seconds)

    def test_adaptive_expands_generation_batch_when_backlogged(self) -> None:
        batched_config = SimulationConfig(
            retrieval=replace(self.config.retrieval, seconds_scale=0.01, static_batch_size=1),
            generation=replace(self.config.generation, seconds_exponent=0.4, static_batch_size=1),
        )
        burst = tuple(Request(index, 0.0) for index in range(12))
        adaptive = run_simulation(burst, batched_config, Policy.PIPELINED_ADAPTIVE)
        self.assertGreater(max(adaptive.generation_batches), 1)


if __name__ == "__main__":
    unittest.main()
