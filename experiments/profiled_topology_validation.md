# Placement-aware topology selection validation

Date: 2026-08-14.  The topology selector used the repeated offline profile to
choose an execution topology for each memory placement, then ran the same
eight-request TriviaQA/Milvus burst once for validation.

| Weight placement | Selected topology | Profiled mean latency | Validation mean latency | Relative error |
| --- | --- | ---: | ---: | ---: |
| 75% GPU / 25% CPU | serial | 2.152 s | 2.168 s | 0.72% |
| 100% GPU / 0% CPU | adaptive | 0.960 s | 0.966 s | 0.63% |

The offloaded placement selected serial execution to avoid retrieval/offload
CPU contention.  The GPU-only placement selected the independently batched,
backlog-aware pipeline.  Both validation measurements are within one percent
of the repeated offline profile, completing the small reproduction's
placement-to-topology decision loop.

This topology choice is a small-model adaptation motivated by observed host
resource contention.  It should not be presented as an exact implementation
detail of the original RAGDoll paper; it operationalizes the paper's broader
profile-driven joint-configuration principle in this constrained setup.
