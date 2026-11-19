---
title: codec
---

::: zarr.abc.codec
    options:
      filters:
        - "!^_[^_]"
        - "!^ArraySpec$"

Codecs receive an `ArraySpec` describing the chunk they encode or decode. The members
listed below are its public API. Its other attributes and its constructor may change
without a deprecation cycle.

::: zarr.abc.codec.ArraySpec
    options:
      members:
        - shape
        - dtype
        - fill_value
        - ndim
        - order
        - prototype
