Unknown members inside a *known* entity's `configuration` (e.g. an extra
key in a `blosc` configuration) now report as their own `unknown_key`
problem kind, and no longer suppress the other rules about that entity.

Whether configurations are closed remains unspecified
([zarr-specs#270](https://github.com/zarr-developers/zarr-specs/issues/270)),
so this package retains its strict reading with two safeguards:

- callers can filter the dedicated `unknown_key` kind;
- unknown keys do not suppress other rules for the same entity.

Model round-trips preserve unmodeled members. Shape-exact `TypeIs` guards
still reject them because the corresponding TypedDicts are closed.
