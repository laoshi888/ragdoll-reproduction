import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_native_partition_budget.py"
SPEC = importlib.util.spec_from_file_location("simulate_native_partition_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_budget_model_rejects_all_resident_and_selects_eight() -> None:
    result = MODULE.evaluate(
        {
            "simulation_label": "simulation",
            "source_experiment": "source",
            "host_memory_budget_gib": 4.45,
            "llm_process_reserve_gib": 4.0,
            "resident_partition_footprint_gib": 0.05,
            "candidate_resident_partitions": [2, 4, 8, 32],
            "observed_end_to_end_latency_seconds": {"2": 5.337, "4": 5.056, "8": 5.019, "32": 3.957},
        }
    )
    assert result["selected_candidate"]["resident_partitions"] == 8
    assert result["candidates"][-1]["within_budget"] is False
