"""Spec transition for the `cast_value` codec.

`cast_value` carries no composition rules of its own today, but it
changes the data type everything downstream receives: a later rule that
reads the type (the `bytes` codec's endianness requirement, for example)
must judge against the configured target.

The codec also casts the fill value, and the spec makes a failed
round-trip a MUST error. Deciding that means implementing the cast
(rounding modes, out-of-range clamp and wrap, scalar maps), which is
numeric semantics rather than JSON judgment; it belongs to whatever
implements the codec and is not modelled here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.rules._spec import ArraySpec, spec_transition
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


@spec_transition(CAST_VALUE_CODEC_NAME)
def cast_data_type(configuration: Mapping[str, object], incoming: ArraySpec) -> ArraySpec:
    """The outgoing type is the configured target."""
    target = cast("ZarrV3MetadataFieldJSON", configuration["data_type"])
    return incoming.with_data_type(target)
