# Logical partition residency profile

Date: 2026-08-15.  We profiled the number of logical Milvus-Lite collections
kept resident between retrieval batches using the same eight-request TriviaQA
burst, OPT-1.3B, and 75% GPU / 25% CPU FlexLLMGen placement.  Each candidate
was run twice with a rotated order.

| Resident partitions | Mean latency (s) | Mean retrieval (s) | Mean loads / releases |
| ---: | ---: | ---: | ---: |
| 1 | 2.983 | 0.313 | 39 / 39 |
| 2 | 2.263 | 0.121 | 36 / 36 |
| 4 | 3.326 | 0.327 | 29 / 29 |
| 8 | **2.167** | **0.088** | **0 / 0** |

The offline selector therefore chooses eight resident partitions for this
small corpus.  This is expected: the full 2,000-vector collection fits easily
in the available host memory, so any lazy reloading adds overhead without
providing a capacity benefit.  The result validates the paper's profiling
principle while also documenting that memory pressure must be introduced by a
larger corpus or tighter host-memory budget before offloading can be expected
to improve latency.
