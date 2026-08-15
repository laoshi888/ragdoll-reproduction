# Partition-residency pressure experiment

The completed eight-partition experiment established that all 2,000 vectors
fit comfortably in host memory: keeping every partition resident avoids all
reloads and is fastest.  This is a correct result for that scale, but it does
not exercise the pressure behind RAGDoll's partition-placement decision.

The paper designs retrieval around a bounded resident database subset, chosen
jointly with model placement and execution scheduling.  The next experiment
keeps the validated OPT-1.3B 75% GPU / 25% CPU FlexLLMGen configuration, while
changing only the corpus/residency axis:

| Variable | Previous profile | Pressure profile |
| --- | ---: | ---: |
| Corpus chunks | 2,000 | 16,000 |
| Logical collections | 8 | 32 |
| Chunks per collection | 250 | 500 |
| Requests in burst | 8 | 16 |
| Resident candidates | 1, 2, 4, 8 | 2, 4, 8, 32 |
| Counterbalanced repetitions | 2 | 3 |

The builder streams TriviaQA and writes a separate source database, then copies
its existing embeddings into logical collections.  It does not alter the
small-corpus artifacts used by the completed result.  Candidate order rotates
between repetitions and the profile records end-to-end latency, retrieval
latency, and load/release counts for every run.

## Interpretation boundary

Milvus Lite has no exposed memory quota and its logical collections are a
compatibility layer rather than native Milvus partitions.  Thus this experiment
measures the transfer/search cost induced by a strict *logical* residency cap;
it cannot by itself prove physical RAM or disk-paging savings.  A result that
32 residents remains fastest means the next necessary escalation is Milvus
Standalone with a measured host-memory limit, not a stronger claim from Lite.

Run `python scripts/check_autodl.py` before construction to inspect the AutoDL
disk capacity.  Then run `bash scripts/run_autodl_partition_pressure.sh` from
the remote project root.  Generated databases and JSON files remain ignored by
Git.
