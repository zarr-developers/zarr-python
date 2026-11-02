"""The rule that the variable-length codecs `vlen-utf8` and `vlen-bytes` share."""

from collections.abc import Iterator

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import AcceptedField
from zarr_metadata.v3._definition import Chunk


def vlen_problems(chunk: Chunk, compatible: frozenset[str]) -> Iterator[ValidationProblem]:
    """Check that the chunk's data type is one of the `compatible` data types, named as their definitions name them.

    Each value is written as its length followed by its bytes, and each
    codec reads those bytes as the values of particular data types, so
    `vlen-utf8` and `vlen-bytes` are not compatible with each other's data
    type even though both vary in size. If the scope did not read the data
    type, nothing is checked.
    """
    data_type = chunk.data_type
    if not isinstance(data_type, AcceptedField) or data_type.definition.name in compatible:
        return
    expected = " or ".join(shown(name) for name in sorted(compatible))
    yield ValidationProblem(
        (),
        f"expected a chunk of data type {expected}, got {shown(data_type.name)}",
        "invalid_value",
    )
