"""The rule that the variable-length codecs `vlen-utf8` and `vlen-bytes` share."""

from collections.abc import Iterator

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import EmptyConfiguration, Nested
from zarr_metadata.v3._definition import Chunk, storage_of


def vlen_chunk_rules(
    configuration: EmptyConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that the data type's values vary in size, the converse of the `bytes` codec's rule.

    Each value is written as its length followed by its bytes, so a data
    type of fixed size needs a fixed-size codec such as `bytes`. The check
    uses the storage that the data type declares, so it does not tell
    `string` from `bytes`. If the storage is unknown, nothing is checked.
    """
    if chunk.data_type is None:
        return
    storage = storage_of(chunk.data_type)
    if storage is not None and storage != "variable_length":
        yield ValidationProblem(
            (),
            f"expected a data type whose values vary in size, got {shown(chunk.data_type.name)}, "
            "whose values have a fixed size",
            "invalid_value",
        )
