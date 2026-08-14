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
environment.  The following two same-model baselines were then run from the
same cached weights and with the same prompt, generation, and batch settings.

| Weight placement (GPU / CPU) | Peak GPU memory | CPU model memory | Total latency | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| 100% / 0% | 2.692 GB | 0.000 GB | 0.631 s | 50.682 token/s |
| 75% / 25% | 1.786 GB | 0.938 GB | 1.503 s | 21.287 token/s |
| 50% / 50% | 1.571 GB | 1.317 GB | 2.004 s | 15.970 token/s |

Relative to the GPU-only baseline, 75%/25% saves 33.7% of peak GPU memory and
50%/50% saves 41.6%, at the cost of lower generation throughput.  The 75%/25%
point improves throughput by 33.3% over 50%/50% for only 13.6% more peak GPU
memory.  This is a component-level memory-placement measurement, not yet the
paper's end-to-end RAGDoll result.

The executable probe accepts `FLEX_MAX_GPU_MEMORY_GIB` and invokes the
profiled selector when explicit `FLEX_PERCENT` is not supplied.  For example,
`FLEX_MAX_GPU_MEMORY_GIB=1.8 bash scripts/run_flexllmgen_opt13b_probe.sh`
selects the 75% GPU / 25% CPU profile above.
