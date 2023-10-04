from typing import Literal, Protocol

ZARR_VERSION = Literal[2, 3]
DIMENSION_SEPARATOR = Literal[".", "/"]
MEMORY_ORDER = Literal["C", "F"]


class MetaArray(Protocol):
    def __array_function__(self, func, types, args, kwargs):
        ...
