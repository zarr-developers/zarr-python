"""
`vlen-utf8` codec types (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-utf8/README.md

The codec has no configuration fields, so the `configuration` key is
optional. The spec's example leaves it out, and zarr-python writes an
empty object, `"configuration": {}`.
"""

from collections.abc import Iterator
from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import Chunk, CodecDefinition
from zarr_metadata.v3._types.codec import (
    VLEN_UTF8_CODEC_NAME,
    VlenUtf8CodecMetadata,
    VlenUtf8CodecName,
    VlenUtf8CodecObject,
)
from zarr_metadata.v3._types.data_type import STRING_DATA_TYPE_NAME
from zarr_metadata.v3.codec._vlen import vlen_problems

_COMPATIBLE: Final = frozenset({STRING_DATA_TYPE_NAME})


def _chunk_rules(
    configuration: EmptyConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that the data type is `string`, as the codec "is only compatible with the `"string"` data type" (https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-utf8/README.md?plain=1#L30)."""
    yield from vlen_problems(chunk, _COMPATIBLE)


VLEN_UTF8_CODEC: Final = CodecDefinition(
    name=VLEN_UTF8_CODEC_NAME,
    configuration=EmptyConfiguration,
    kind="array_bytes",
    size="dynamic",
    chunk_rules=_chunk_rules,
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
