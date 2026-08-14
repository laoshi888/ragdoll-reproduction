# Qwen 1.5B burst ablation (vLLM integration)

This is an intermediate reproducibility result for the vLLM-backed pipeline,
not a claim of full RAGDoll reproduction. It uses the small TriviaQA/Milvus
corpus, Qwen/Qwen2.5-1.5B-Instruct, and the 64-request burst configuration in
`configs/real_burst.yaml` on the AutoDL RTX 4090.

| Mode | Mean end-to-end latency (s) | P95 latency (s) | Mean waiting (s) | Mean retrieval (s) | Mean generation (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 2.499 | 3.529 | 1.041 | 0.035 | 1.424 |
| Pipelined static | 2.147 | 2.822 | 0.745 | 0.027 | 1.374 |
| Pipelined adaptive | 2.283 | 3.221 | 0.912 | 0.027 | 1.344 |

The pipelined static mode reduces mean latency by 14.1% relative to the
serial mode, driven primarily by a 28.4% reduction in waiting time. Under this
small-model configuration, the current adaptive policy does not outperform the
fixed policy. This is expected to be revisited only after adding a controllable
offloading backend and memory-placement configuration.
