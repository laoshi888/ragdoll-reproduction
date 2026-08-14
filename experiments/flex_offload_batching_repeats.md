# Repeated FlexLLMGen offload batching experiment

Date: 2026-08-14.  Three counterbalanced rounds ran the same eight-request
TriviaQA/Milvus burst using the 75% GPU / 25% CPU OPT-1.3B weight placement.
The generation worker supported batches of one or two requests.  Profiles were
warm-started before collecting the three timing samples per batch size.

| Policy | Mean end-to-end | Median end-to-end | Mean P95 | Mean waiting | Mean retrieval | Mean generation | Mean wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 2.152 s | 2.145 s | 3.278 s | 1.347 s | 0.053 s | 0.752 s | 4.099 s |
| static (B=2) | 2.655 s | 2.246 s | 3.876 s | 1.701 s | 0.091 s | 0.863 s | 4.700 s |
| adaptive (B in {1,2}) | 3.111 s | 3.202 s | 4.446 s | 2.035 s | 0.134 s | 0.943 s | 5.270 s |

The simple isolated-stage profiles correctly show that generation batch two
has almost the same total runtime as batch one.  However, the two-worker
pipeline overlaps CPU retrieval with FlexLLMGen CPU weight offloading.  The
observed generation time rises from 0.752 s in serial execution to 0.863 s
for static and 0.943 s for adaptive.  Thus this small offload configuration is
CPU-contention limited: independent per-stage profiles are insufficient to
predict the cost of overlap.  A paired GPU-only experiment is needed to
isolate the offload-induced contention from general scheduling overhead.
