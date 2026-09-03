"""Classify Zarr v3 codecs by pipeline kind.

The v3 spec sorts codecs into three kinds — `array -> array`,
`array -> bytes`, `bytes -> bytes` — and a pipeline is
`array->array* array->bytes bytes->bytes*`. `codec_kind_of_name`
classifies a known name; unknown names have no kind.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC_NAME
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC_NAME
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME

ARRAY_ARRAY_CODEC_NAMES: Final = (
    TRANSPOSE_CODEC_NAME,
    CAST_VALUE_CODEC_NAME,
    SCALE_OFFSET_CODEC_NAME,
)
"""Tuple of the `name` field values of the known `array -> array` codecs."""

ARRAY_BYTES_CODEC_NAMES: Final = (BYTES_CODEC_NAME, SHARDING_INDEXED_CODEC_NAME)
"""Tuple of the `name` field values of the known `array -> bytes` codecs."""

BYTES_BYTES_CODEC_NAMES: Final = (
    BLOSC_CODEC_NAME,
    CRC32C_CODEC_NAME,
    GZIP_CODEC_NAME,
    ZSTD_CODEC_NAME,
)
"""Tuple of the `name` field values of the known `bytes -> bytes` codecs."""

CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""The three pipeline positions the v3 spec sorts codecs into."""


def codec_kind_of_name(name: str) -> CodecKind | None:
    """The pipeline kind of the codec named `name`, or None if unknown.

    Classifies by name alone, with no judgment of the entry's spelling or
    configuration; the rules layer judges those separately.
    """
    if name in ARRAY_ARRAY_CODEC_NAMES:
        return "array_array"
    if name in ARRAY_BYTES_CODEC_NAMES:
        return "array_bytes"
    if name in BYTES_BYTES_CODEC_NAMES:
        return "bytes_bytes"
    return None


__all__ = [
    "ARRAY_ARRAY_CODEC_NAMES",
    "ARRAY_BYTES_CODEC_NAMES",
    "BYTES_BYTES_CODEC_NAMES",
    "CodecKind",
    "codec_kind_of_name",
]
