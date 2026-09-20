"""Composition rules for the core `zstd` codec.

The spec gives `level` a range the shape validator cannot state: "An
integer from -131072 to 22 which controls the speed and level of
compression (has no impact on decoding). A value of 0 indicates to use the
default compression level."

`checksum` needs no rule: the spec marks it "(Optional)" and the TypedDict
already declares it `NotRequired`. Its "Should be omitted if false" is a
SHOULD, and this package reports violations of requirements rather than of
advice.

https://github.com/zarr-developers/zarr-extensions/blob/main/codecs/zstd/README.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArrayParts

_ARRAY_V3 = "zarr_v3_array"
_MIN_LEVEL: Final = -131072
_MAX_LEVEL: Final = 22


@entity_rule(_ARRAY_V3, CODECS, ZSTD_CODEC_NAME, reads=frozenset({"level"}))
def level_is_in_range(
    configuration: Mapping[str, object],
    document: Mapping[str, object],
    incoming: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    level = cast("int", configuration["level"])
    if _MIN_LEVEL <= level <= _MAX_LEVEL:
        return ()
    return (
        ValidationProblem(
            ("level",),
            f"expected an integer in [{_MIN_LEVEL}, {_MAX_LEVEL}], got {level}",
            "invalid_value",
        ),
    )
