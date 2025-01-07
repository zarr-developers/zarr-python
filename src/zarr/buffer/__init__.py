from zarr.abc.buffer import BufferPrototype
from zarr.buffer.cpu import numpy_buffer_prototype

__all__ = [
    "default_buffer_prototype",
    "numpy_buffer_prototype",
]


# The default buffer prototype used throughout the Zarr codebase.
def default_buffer_prototype() -> BufferPrototype:
    from zarr.registry import (
        get_buffer_class,
        get_ndbuffer_class,
    )

    return BufferPrototype(buffer=get_buffer_class(), nd_buffer=get_ndbuffer_class())
