# Logical partition residency design

The paper treats the number of resident vector-database partitions as a joint
memory-placement parameter.  Partitions are loaded or released lazily between
retrieval batches; offline profiling then balances retrieval latency against
generation latency and host-memory pressure.

This small reproduction uses Milvus Lite, whose official feature matrix does
not support native partitions or partition-scoped search.  To preserve the
paper's control point without requiring Docker or a second database service,
each logical partition is represented by a separate Milvus Lite collection.
The retriever:

1. embeds each query batch once;
2. searches all logical partitions and merges a global top-k, preserving exact
   retrieval over the 2,000-chunk corpus;
3. keeps a configurable number of the strongest-hit partitions loaded between
   batches; and
4. lazily loads and releases non-resident collections while recording transfer
   and search counts.

The partitioned database is built by copying the existing vectors in batches
to a separate database file.  The source collection is never dropped or
re-embedded.  This is a Milvus-Lite compatibility implementation of the
paper's partition-residency control, not a claim that Lite provides native
GPU/CPU/disk partition placement.  A later Milvus Standalone deployment can
replace the collection mapping with native partitions while retaining the
same configuration and scheduler interface.
