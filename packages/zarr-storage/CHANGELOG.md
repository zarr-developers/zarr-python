# zarr-storage changelog

## Unreleased

- Extract the existing storage interfaces, concrete implementations, and accessory
  stores into `zarr_storage.legacy`, preserving Zarr's current runtime imports.
- Include the store and experimental cache-store suites and distribute reusable
  conformance tests and stateful testing utilities in `zarr_storage.testing`.
- Make `LatencyStore` usable without importing pytest.
