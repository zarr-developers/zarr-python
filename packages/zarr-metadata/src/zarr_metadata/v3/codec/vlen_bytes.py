"""
`vlen-bytes` codec types (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-bytes/README.md

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
    VLEN_BYTES_CODEC_NAME,
    VlenBytesCodecMetadata,
    VlenBytesCodecName,
    VlenBytesCodecObject,
)
from zarr_metadata.v3._types.data_type import (
    BYTES_DATA_TYPE_NAME,
    VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME,
)
from zarr_metadata.v3.codec._vlen import vlen_problems

_COMPATIBLE: Final = frozenset({BYTES_DATA_TYPE_NAME, VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME})


def _chunk_rules(
    configuration: EmptyConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that the data type is `bytes`, as the codec "is only compatible with the `"bytes"` data type" (https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-bytes/README.md?plain=1#L30).

    zarr-python's `variable_length_bytes` holds the same values under
    another name, so it is compatible too. Only `ZARR_PYTHON_CONTEXT` reads
    that data type.
    """
    yield from vlen_problems(chunk, _COMPATIBLE)


VLEN_BYTES_CODEC: Final = CodecDefinition(
    name=VLEN_BYTES_CODEC_NAME,
    configuration=EmptyConfiguration,
    kind="array_bytes",
    size="dynamic",
    chunk_rules=_chunk_rules,
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
