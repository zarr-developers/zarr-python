"""The v2 variable-length filters that numcodecs 0.16 configures: `vlen-utf8`, `vlen-bytes` and `vlen-array`."""

from __future__ import annotations

from typing import Final

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.definition._definition import EmptyConfiguration
from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2.codec._dtype import dtype_parameter


class ZarrV2VLenArrayParameters(TypedDict, closed=True):
    """`numcodecs.VLenArray(dtype)`: the element type of each array."""

    dtype: ReadOnly[str]


VLEN_UTF8_V2: Final = ZarrV2CodecDefinition(name="vlen-utf8", configuration=EmptyConfiguration)
"""`numcodecs.VLenUTF8`. It has no configuration."""
VLEN_BYTES_V2: Final = ZarrV2CodecDefinition(name="vlen-bytes", configuration=EmptyConfiguration)
"""`numcodecs.VLenBytes`. It has no configuration."""
VLEN_ARRAY_V2: Final = ZarrV2CodecDefinition(
    name="vlen-array", configuration=ZarrV2VLenArrayParameters, rules=dtype_parameter("dtype")
)
"""`numcodecs.VLenArray`."""

__all__ = ["VLEN_ARRAY_V2", "VLEN_BYTES_V2", "VLEN_UTF8_V2", "ZarrV2VLenArrayParameters"]
