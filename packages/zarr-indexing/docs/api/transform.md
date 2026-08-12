---
title: transform
---

An `IndexTransform` is a function between coordinate spaces, and its field
names follow the function, not the data:

| the API says | in array terms |
| --- | --- |
| input space (`domain`, `input_rank`) | request coordinates — the result being built |
| output space (`output`, one map per dimension) | source coordinates — where values are read |

`output` is not data: it is the rule, per source dimension, for producing
coordinates. Values flow source → request, against the arrow. The
[guide](../guide/index.md#a-transform-points-from-the-request-to-the-source)
demonstrates each output map form against its NumPy counterpart.

::: zarr_indexing.transform
