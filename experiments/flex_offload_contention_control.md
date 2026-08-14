# GPU-only control for the FlexLLMGen offload batching experiment

Date: 2026-08-14.  This is the GPU-only counterpart of the repeated offload
experiment.  It used the same OPT-1.3B model, TriviaQA/Milvus corpus,
eight-request burst, prompt/generation lengths, batch candidates, warm-up,
and counterbalanced policy order.  The only changed variable was the profiled
weight placement: 100% GPU rather than 75% GPU / 25% CPU.

| Policy | 75/25 offload mean latency | GPU-only mean latency | GPU-only P95 | GPU-only mean generation | GPU-only wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial | 2.152 s | 1.595 s | 2.257 s | 0.510 s | 3.080 s |
| static (B=2) | 2.655 s | 1.897 s | 2.456 s | 0.562 s | 3.280 s |
| adaptive (B in {1,2}) | 3.111 s | 0.960 s | 1.111 s | 0.326 s | 1.933 s |

With GPU-only weights, adaptive is the fastest policy: it improves mean
end-to-end latency by 39.8% versus serial and 49.4% versus static.  In the
75/25 offload condition, adaptive is 3.24x slower than its GPU-only control.
This isolates the reversal observed in the offload experiment: CPU retrieval
(embedding and Milvus Lite) contends with the CPU traffic introduced by
offloaded model weights.  The current independent-stage selector does not
profile this overlap cost, so it cannot choose the right pipeline topology for
the offloaded placement.
