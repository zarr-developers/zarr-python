"""
`vlen-utf8` codec types (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-utf8/README.md

The codec has no configuration fields, so the `configuration` key is
absent from the metadata.
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.definition._definition import EmptyConfiguration
from zarr_metadata.v3._definition import CodecDefinition
from zarr_metadata.v3.codec._vlen import vlen_chunk_rules

VLEN_UTF8_CODEC_NAME: Final = "vlen-utf8"
"""The `name` field value of the `vlen-utf8` codec."""

VlenUtf8CodecName = Literal["vlen-utf8"]
"""Literal type of the `name` field of the `vlen-utf8` codec."""


class VlenUtf8CodecObject(TypedDict, closed=True):
    """`vlen-utf8` codec metadata in object form. The codec has no configuration fields."""

    name: VlenUtf8CodecName
    configuration: NotRequired[EmptyConfiguration]
    must_understand: NotRequired[bool]


VlenUtf8CodecMetadata = VlenUtf8CodecObject | VlenUtf8CodecName
"""Permitted JSON shapes for `vlen-utf8` codec metadata: the object form, or the bare name, since the codec has no configuration."""


VLEN_UTF8_CODEC: Final = CodecDefinition(
    name=VLEN_UTF8_CODEC_NAME,
    configuration=EmptyConfiguration,
    kind="array_bytes",
    size="dynamic",
    chunk_rules=vlen_chunk_rules,
)
"""The `vlen-utf8` codec, which serializes variable-length UTF-8 strings.

It is "only compatible with the `"string"` data type"
(https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-utf8/README.md?plain=1#L30).
The size of its output depends on the values, so it is dynamic.
"""

__all__ = [
    "VLEN_UTF8_CODEC",
    "VLEN_UTF8_CODEC_NAME",
    "VlenUtf8CodecMetadata",
    "VlenUtf8CodecName",
    "VlenUtf8CodecObject",
]
