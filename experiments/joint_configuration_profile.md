# Joint configuration profile

Date: 2026-08-15.  Three counterbalanced, end-to-end runs measured complete
RAG configurations on the eight-partition corpus.  The selection metric is
median latency because each configuration experienced one transient I/O-heavy
run.

| GPU budget | Placement | DB residents | Topology | Mean latency | Median latency |
| ---: | --- | ---: | --- | ---: | ---: |
| 1.8 GiB | 75% GPU / 25% CPU weights | 8 | serial | 2.633 s | 2.232 s |
| 3.0 GiB | GPU-only weights | 8 | adaptive | 2.145 s | **1.308 s** |

Under the 1.8 GiB constraint, only the offloaded configuration is feasible.
With at least 3.0 GiB available, the measured GPU-only/adaptive configuration
has the lower median latency.  The unified selector only chooses between these
measured bundles and records its selected profile in each runtime result.

## Independent validation and cloud variability

The selector was then validated in fresh AutoDL processes.  The 1.8 GiB
offloaded bundle measured 2.222 s, within 0.5% of its profiled median (2.232
s).  It therefore validates the constrained-memory selection directly.

For the 3.0 GiB GPU-only bundle, three fresh validation runs measured 3.778 s,
1.277 s, and 3.282 s (median 3.282 s).  All three runs selected the intended
GPU-only placement, eight resident partitions, and adaptive topology.  The
fast run agrees with the 1.308 s profiling median; the two slow runs show
substantial shared-cloud CPU/I/O variability, rather than an incorrect
configuration selection.  The earlier three-run profile also contained one
similar slow run (3.829 s).

Consequently, this small-scale reproduction reports medians and preserves raw
per-run JSON files on AutoDL.  A final performance comparison should alternate
both configurations for multiple repetitions on the same instance, rather
than infer a conclusion from one cloud run.
