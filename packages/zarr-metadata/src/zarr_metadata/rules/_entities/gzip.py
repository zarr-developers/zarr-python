"""Composition rules for the core ``gzip`` codec."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, CODECS, GZIP_CODEC_NAME)
def level_is_in_range(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    level = cast("int", configuration["level"])
    if 0 <= level <= 9:
        return ()
    return (
        ValidationProblem(
            ("level",), f"expected an integer in [0, 9], got {level}", "invalid_value"
        ),
    )
