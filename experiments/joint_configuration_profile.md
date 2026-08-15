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
