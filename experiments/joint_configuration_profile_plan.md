# Joint configuration profile plan

The next active-profile step measures complete configurations rather than
selecting generation placement, database residency, and execution topology in
isolation.  It uses the same eight-request partitioned TriviaQA burst and three
counterbalanced repetitions per candidate.  Existing isolated run files are
reused, so interrupted or extended profiles execute only missing repetitions.

| Candidate | GPU budget | Expected measured placement | Resident DB partitions | Profiled topology |
| --- | ---: | --- | ---: | --- |
| `offload_75_25` | 1.8 GiB | 75% GPU / 25% CPU weights | 8 | serial |
| `gpu_only` | 3.0 GiB | 100% GPU weights | 8 | adaptive |

The expectations above come from the existing component profiles; the joint
run records the actual placement, residency, topology, waiting time,
retrieval time, generation time, and end-to-end latency.  These measurements
will become the input to the unified configuration selector.  No unmeasured
cross-product configuration is inferred.
