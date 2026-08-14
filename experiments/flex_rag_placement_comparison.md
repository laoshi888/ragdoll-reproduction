# FlexLLMGen end-to-end RAG placement comparison

Date: 2026-08-14.  The same four-request TriviaQA/Milvus workload was run with
the same prompt length (128), generation length (16), and batch size (1).
Only the offline-profiled FlexLLMGen weight placement changed.

| Placement | Peak GPU memory (profiled) | Mean end-to-end | P95 end-to-end | Mean generation | Mean waiting | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 75% GPU / 25% CPU | 1.786 GB | 1.776 s | 2.538 s | 0.741 s | 0.959 s | 3.184 s |
| 100% GPU / 0% CPU | 2.692 GB | 0.887 s | 1.052 s | 0.365 s | 0.433 s | 1.700 s |

The offloaded placement saves 33.7% of peak GPU memory, while its mean
end-to-end latency is 2.00x the GPU-only baseline and its mean generation
time is 2.03x.  Waiting time also rises because slower generation leaves more
items queued.  This is an end-to-end RAG result: every request first retrieves
three real Milvus contexts before running the selected FlexLLMGen placement.
