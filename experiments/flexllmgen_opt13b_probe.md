# FlexLLMGen OPT-1.3B memory-placement probe

Date: 2026-08-14.  Hardware: one RTX 4090 (24 GB).  The command was
`bash scripts/run_flexllmgen_opt13b_probe.sh` with the default placement
`50 50 100 0 100 0`: weights 50% GPU / 50% CPU, KV cache fully GPU, and
activations fully GPU.  Prompt length was 128, generated length was 32, and
the GPU batch size and number of GPU batches were both 1.

| Metric | Result |
| --- | ---: |
| Model weight size | 2.443 GB |
| GPU resident model memory | 1.333 GB |
| CPU resident model memory | 1.317 GB |
| Peak GPU memory | 1.571 GB |
| Prefill latency | 0.060 s |
| Decode latency | 1.943 s |
| Total latency | 2.004 s |
| Decode throughput | 15.952 token/s |

This confirms that FlexLLMGen's CPU/GPU weight placement works in the AutoDL
environment.  It is a component-level measurement, not yet the paper's
end-to-end RAGDoll result.  The next controlled run uses `FLEX_PERCENT="100 0
100 0 100 0"` to establish the same-model GPU-only baseline.
