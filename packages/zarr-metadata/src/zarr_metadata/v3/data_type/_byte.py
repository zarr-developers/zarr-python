"""A byte value, as used in the fill values of raw bits and of `bytes`.

Each is an integer in `[0, 255]`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99;
the reference for the `bytes` type is https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L8).
"""

from typing import Annotated

from annotated_types import Interval

ByteValue = Annotated[int, Interval(ge=0, le=255)]
"""One byte of a fill value: a JSON integer in `[0, 255]`."""

__all__ = ["ByteValue"]
