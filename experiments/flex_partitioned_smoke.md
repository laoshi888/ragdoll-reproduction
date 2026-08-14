# Logical partition residency smoke test

Date: 2026-08-14.  The existing 2,000-vector TriviaQA-derived Milvus Lite
collection was copied into eight logical collections (250 vectors each) and
verified without rebuilding embeddings.  The source collection remained
unchanged.

Configuration: OPT-1.3B with the 75% GPU / 25% CPU FlexLLMGen weight
placement, two resident logical partitions, eight requests, serial topology,
and retrieval/generation batch sizes selected from 1 and 2.

| Metric | Result |
| --- | ---: |
| Mean end-to-end latency | 2.624 s |
| P95 latency | 3.936 s |
| Mean waiting time | 1.680 s |
| Mean retrieval time | 0.158 s |
| Mean generation time | 0.785 s |
| Logical partition searches | 40 |
| Lazy loads / releases | 36 / 36 |

The result validates the Milvus-Lite compatibility path: all eight logical
collections are searched and merged into a global top-k while only the hot
subset remains resident between batches.  The next offline profiling run
compares resident counts 1, 2, 4, and 8 using two counterbalanced repetitions
per candidate.
