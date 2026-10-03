"""
URL pipeline adapters shipped with zarr-python.

These are registered as lazily loaded builtin entry points by
`zarr.registry` (see `_BUILTIN_URL_ADAPTERS`), so nothing in this package is
imported until a pipeline naming one of its schemes is resolved.
"""
