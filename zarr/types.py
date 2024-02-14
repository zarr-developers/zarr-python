from typing import Literal, Protocol, Union

ZARR_VERSION = Literal[2, 3]
DIMENSION_SEPARATOR = Literal[".", "/"]
MEMORY_ORDER = Literal["C", "F"]


PathLike = Union[str, bytes, None]


class MetaArray(Protocol):
    def __array_function__(self, func, types, args, kwargs):
        # To be extended
        ...
