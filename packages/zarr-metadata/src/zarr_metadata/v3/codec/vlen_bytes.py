"""
`vlen-bytes` codec types (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-bytes/README.md

The codec has no configuration fields, so the `configuration` key is
absent from the metadata.
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.definition._definition import EmptyConfiguration
from zarr_metadata.v3._definition import CodecDefinition
from zarr_metadata.v3.codec._vlen import vlen_chunk_rules

VLEN_BYTES_CODEC_NAME: Final = "vlen-bytes"
"""The `name` field value of the `vlen-bytes` codec."""

VlenBytesCodecName = Literal["vlen-bytes"]
"""Literal type of the `name` field of the `vlen-bytes` codec."""


class VlenBytesCodecObject(TypedDict, closed=True):
    """`vlen-bytes` codec metadata in object form. The codec has no configuration fields."""

    name: VlenBytesCodecName
    configuration: NotRequired[EmptyConfiguration]
    must_understand: NotRequired[bool]


VlenBytesCodecMetadata = VlenBytesCodecObject | VlenBytesCodecName
"""Permitted JSON shapes for `vlen-bytes` codec metadata: the object form, or the bare name, since the codec has no configuration."""


VLEN_BYTES_CODEC: Final = CodecDefinition(
    name=VLEN_BYTES_CODEC_NAME,
    configuration=EmptyConfiguration,
    kind="array_bytes",
    size="dynamic",
    chunk_rules=vlen_chunk_rules,
)
"""The `vlen-bytes` codec, which serializes variable-length byte strings.

It is "only compatible with the `"bytes"` data type"
(https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-bytes/README.md?plain=1#L30).
The size of its output depends on the values, so it is dynamic.
"""

__all__ = [
    "VLEN_BYTES_CODEC",
    "VLEN_BYTES_CODEC_NAME",
    "VlenBytesCodecMetadata",
    "VlenBytesCodecName",
    "VlenBytesCodecObject",
]
