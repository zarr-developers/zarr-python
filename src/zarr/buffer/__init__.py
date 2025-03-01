"""
Public API for implementations of the Zarr Buffer interface.

See Also
========
arr.abc.buffer: Abstract base class for the Zarr Buffer interface.
"""

from ..core.buffer import default_buffer_prototype
from . import cpu, gpu

__all__ = ["cpu", "default_buffer_prototype", "gpu"]
