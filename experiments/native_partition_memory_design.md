# Native partition and constrained-memory design

The logical-collection experiment demonstrated the cost of repeated loads and
releases, but Milvus Lite cannot enforce native partition residency or a main-
memory limit.  This stage implements the remaining paper control point with
Milvus Standalone v2.5.11.

RAGDoll bounds CPU placement with

`LLM weights in CPU + KV cache in CPU + P * partition size <= CPU memory`,

where `P` is the number of resident database partitions.  It changes `P` by
loading and releasing partitions lazily between retrieval batches, then uses
offline measurements to select a feasible configuration that balances
retrieval and generation latency.

## Reproduction mapping

| Paper mechanism | Small reproduction |
| --- | --- |
| One vector collection with 32 on-disk partitions | One Milvus Standalone collection with 32 native partitions |
| Lazy partition load/release | `load_partitions` and `release_partitions` between retrieval batches |
| Exact retrieval across the knowledge base | Search every partition and merge a global top-k |
| Bounded CPU memory | 4 GiB Docker cgroup limit with swap disabled for Milvus |
| Joint host-memory feasibility | Milvus usage plus LLM-process peak RSS must fit an 8 GiB experiment budget |
| Offline configuration exploration | Three rotated repetitions for 2, 4, 8, 16, and 32 residents |

The corpus reuses the existing 16,000 TriviaQA embeddings.  Vectors remain in
their original modulo-assigned partition and are repeated to 32,768 rows per
partition, yielding 1,048,576 rows total.  This avoids another dataset or model
download and creates roughly 1.5 GiB of raw vector payload before index, text,
and service overhead.  Repetition is suitable for placement and latency
pressure, but the result must not be interpreted as a retrieval-quality
measurement.

Index construction receives an 8 GiB temporary container limit.  After the
collection is flushed, indexed, and released, the runner lowers the live Milvus
container to 4 GiB and sets the memory-plus-swap limit to the same value.  The
profiler verifies the cgroup limit before accepting any candidate.  A failed
load/search or an OOM-killed container marks that candidate infeasible instead
of terminating the entire search.

The official Milvus documentation defines native partition-scoped load,
release, and search APIs and documents Docker memory limits for Standalone:

- https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Partitions/load_partitions.md
- https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Partitions/release_partitions.md
- https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/search.md
- https://milvus.io/docs/scale-standalone.md

The first AutoDL step is read-only preflight.  It must confirm Docker access,
Compose support, cgroup availability, at least 10 GiB free disk, and adequate
host memory before any images or corpus artifacts are created.
