# Simulated native-partition budget selection

This is an analytical simulation, not a native Milvus measurement. It closes
the configuration-selection loop that could not be executed on the current
AutoDL container because that environment has no Docker daemon or cgroup
administration privilege.

The model follows the paper's placement constraint:

`LLM reserve + resident partitions * estimated partition footprint <= memory budget`.

The candidate latencies are calibrated from the completed 16k logical
partition-pressure experiment, while the memory values below are explicit
assumptions for a synthetic 4.45 GiB host budget: 4.00 GiB reserved for the
LLM process and 0.05 GiB per resident partition. This budget makes 32
residents infeasible; it is not a measurement of the AutoDL host.

| Resident partitions | Estimated total memory | Calibrated end-to-end latency | Feasible |
| ---: | ---: | ---: | :--- |
| 2 | 4.10 GiB | 5.337 s | yes |
| 4 | 4.20 GiB | 5.056 s | yes |
| 8 | 4.40 GiB | **5.019 s** | yes — selected |
| 32 | 5.60 GiB | 3.957 s | no |

Therefore the simulated controller selects eight resident partitions: it is
the lowest-latency candidate that fits the modeled budget. The result
demonstrates the constraint-aware selection logic only. It must not be
interpreted as proof of physical memory savings, native `load_partitions` /
`release_partitions` behavior, or a Docker cgroup limit.

Reproduce this tiny calculation locally with:

```bash
python scripts/simulate_native_partition_budget.py
```
