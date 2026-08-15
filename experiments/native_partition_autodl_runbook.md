# AutoDL runbook: native partitions under a real RAM cap

All local work stops at source, configuration, and dependency-free tests.
Docker, Milvus Standalone, corpus construction, cgroup enforcement, and timing
measurements run only at `/root/autodl-tmp/ragdoll` on AutoDL.

## Gate 1: read-only preflight

After the relevant commit has been pulled, run only:

```bash
cd /root/autodl-tmp/ragdoll
bash scripts/check_native_milvus_autodl.sh
```

This prints disk, host memory, Docker/Compose versions, and cgroup mode.  It
does not pull images, start containers, or create data.  Do not continue unless
it ends with `native_milvus_preflight=passed`.

## Gate 2: native smoke

Run:

```bash
bash scripts/run_autodl_native_partition_smoke.sh
```

This starts pinned Milvus Standalone v2.5.11, builds only 16,000 rows in 32
native partitions, applies the 4 GiB cgroup limit with swap disabled, and
profiles resident counts 2 and 32 once each.  Required evidence is:

- 32 native partitions with 500 rows each and `vector_ivf` verified;
- container memory and memory-swap limits both equal 4 GiB;
- both profile runs complete with `feasible: true`;
- native load/release counters are non-zero for two residents and zero for 32.

Stop and diagnose at this gate if any API, Docker, cgroup, or memory measurement
does not match those expectations.

## Gate 3: full constrained-memory profile

Only after Gate 2 succeeds, run:

```bash
bash scripts/run_autodl_native_partition_profile.sh
```

The full builder reuses the same 16,000 source embeddings and repeats them
within their assigned native partitions to reach 1,048,576 rows.  It builds an
IVF_FLAT index with temporary 8 GiB construction headroom, releases the
collection, then lowers Milvus to the measured 4 GiB runtime cap.  The profiler
runs resident counts 2, 4, 8, 16, and 32 three times in rotated order.

A candidate is selectable only when the pipeline succeeds, the Milvus
container is not OOM-killed, and Milvus usage plus the LLM process's peak RSS
does not exceed the configured 8 GiB host budget.  Failed candidates remain in
`infeasible_runs`; a cgroup OOM triggers container recovery before the next
candidate.  Selection uses median end-to-end latency among feasible candidates.

Generated credentials, service volumes, collection markers, and raw JSON files
remain under ignored `data/` or `experiments/` paths and must not be committed.

After results have been copied, release container memory without deleting the
persisted database:

```bash
bash scripts/stop_native_milvus.sh
```
