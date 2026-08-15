# Logical partition-residency pressure profile

Date: 2026-08-15.  We increased the TriviaQA-derived corpus from 2,000 to
16,000 chunks and represented it as 32 equal Milvus-Lite logical collections
(500 vectors each).  With OPT-1.3B and the validated 75% GPU / 25% CPU
FlexLLMGen placement, a 16-request burst profiled resident counts 2, 4, 8,
and 32.  Three counterbalanced repetitions were run per candidate.

| Resident partitions | Mean end-to-end latency | Mean retrieval time | Mean loads / releases |
| ---: | ---: | ---: | ---: |
| 2 | 5.337 s | 0.560 s | 285 / 285 |
| 4 | 5.056 s | 0.483 s | 282 / 282 |
| 8 | 5.019 s | 0.445 s | 266 / 266 |
| 32 | **3.957 s** | **0.193 s** | **0 / 0** |

At this scale, keeping every partition resident reduces mean end-to-end
latency by 1.35x relative to two residents (5.337 s to 3.957 s) and reduces
mean retrieval time by 2.90x.  Smaller resident sets incur repeated logical
collection loads and releases: 285 per direction for two residents across the
16-request burst.  All candidates used the same retrieval/generation batch
sequence and measured GPU placement.

This is a successful pressure validation of the residency control path: the
number of transfers grows sharply as the resident cap tightens, and its cost is
visible in end-to-end latency.  It does **not** establish that evicting
partitions saves physical host memory, because Milvus Lite has no exposed
memory quota and uses logical collections as a compatibility mapping.  As 32
residents is still fastest, the next experiment should use Milvus Standalone
with a measured host-memory constraint or an otherwise enforceable native
partition budget.

Raw per-run JSON files are kept under `experiments/partition_residency_runs/`
on AutoDL and are intentionally ignored by Git.
