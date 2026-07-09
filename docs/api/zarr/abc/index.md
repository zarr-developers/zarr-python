---
title: zarr.abc
---

# zarr.abc

Abstract base classes for extending Zarr-Python.

- **[zarr.abc.buffer](./buffer.md)** - Providing access to underlying memory via [buffers](https://docs.python.org/3/c-api/buffer.html)
- **[zarr.abc.codec](./codec.md)** - Expressing [zarr codecs](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding)
- **[zarr.abc.metadata](./metadata.md)** - Creating metadata classes compatible with the Zarr API
- **[zarr.abc.numcodec](./numcodec.md)** - Protocols and classes for modeling codec interface used by numcodecs
- **[zarr.abc.store](./store.md)** - ABC for implementing Zarr stores and managing getting and setting bytes in a store
