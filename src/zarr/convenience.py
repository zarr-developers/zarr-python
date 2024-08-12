import warnings

warnings.warn(
    "zarr.convenience is deprecated, use zarr.api.synchronous",
    DeprecationWarning,
    stacklevel=2,
)
