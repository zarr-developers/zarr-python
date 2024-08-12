import warnings

warnings.warn(
    "zarr.creation is deprecated, use zarr.api.synchronous",
    DeprecationWarning,
    stacklevel=2,
)
