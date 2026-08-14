# FlexLLMGen end-to-end RAG smoke test

Date: 2026-08-14.  This run used the small TriviaQA/Milvus corpus and four
requests at 120 requests per minute.  The FlexLLMGen generator loaded
`facebook/opt-1.3b` with the profiled 75% GPU / 25% CPU weight placement;
KV cache and activations remained on the GPU.  Each request retrieved three
contexts and generated up to 16 tokens.

| Metric | Result |
| --- | ---: |
| Completed requests | 4 |
| Mean end-to-end latency | 1.881 s |
| P95 end-to-end latency | 2.662 s |
| Mean queueing time | 1.023 s |
| Mean retrieval time | 0.091 s |
| Mean generation time | 0.767 s |
| Whole workload wall time | 3.309 s |

This validates the complete small-model path: real retrieval context is
tokenized, passed to the profiled FlexLLMGen offloading generator, and mapped
back to the original request IDs.  A GPU-only run with the same workload is
the next controlled comparison.
