# zarr-metadata

Spec-defined metadata types for Zarr v2 and v3, distributed as pure-typing
artifacts (TypedDicts, type aliases, unions). No runtime logic, no numpy,
no storage backends.

`zarr-metadata` is developed in the [zarr-python](https://github.com/zarr-developers/zarr-python)
repository at `packages/zarr-metadata/`.

## Principle

Every type that models a spec artifact (v2 or v3 array/group/consolidated
metadata, chunk grids, codec named-config envelopes, dtype shapes) belongs
in `zarr-metadata`. Zarr-python implementation details (runtime codecs,
config dataclasses, numcodecs-derived helpers) stay in `zarr`.
