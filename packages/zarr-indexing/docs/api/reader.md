---
title: Readers
---

# Readers

An `IndexTransform` defines which source value belongs at every result
position. A `Reader` defines how a particular backend obtains those values.
Readers do not define indexing semantics, partitioning, scheduling, or result
ownership.

::: zarr_indexing.reader
