import warnings

from zarr.abc.buffer import Buffer, BufferPrototype, NDBuffer  # noqa: F401

warnings.warn(
    "The zarr.core.buffer module is deprecated and will be removed in a future version. "
    "Please use zarr.buffer and zarr.abc.buffer modules instead.",
    FutureWarning,
    stacklevel=2,
)
