"""Spec transition for the `scale_offset` codec.

`scale_offset` subtracts an offset and multiplies by a scale, element by
element, and the spec requires the result to be representable in the
array's own data type: "The encoding and decoding transformations MUST be
performed using the arithmetic semantics of the input array's data type.
If any intermediate or final value produced during encoding or decoding
is not representable in that data type, implementations MUST treat this
as an error."

So it changes neither the shape nor the data type, and its transition is
the identity. The spec is explicit that narrowing is somebody else's job:
the codec's `astype` field "was removed from the `scale_offset` codec in
favor of expressing data type conversion via a dedicated codec", because
"in Zarr V3, a `dtype` field is not needed — the data type of the input
to an array-array codec is determined by its location in the `codecs`
metadata".

Registering this matters beyond tidiness. A modelled `array -> array`
codec with no transition is treated as unknown, which stops propagation
and silently stands down every rule downstream of it — so without this,
inserting a no-op `scale_offset` would switch off the `bytes` codec's
endianness requirement and every shard's geometry check.

https://github.com/zarr-developers/zarr-extensions/blob/main/codecs/scale_offset/README.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.rules._spec import spec_transition
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArrayParts


@spec_transition(SCALE_OFFSET_CODEC_NAME)
def preserves_the_array(configuration: Mapping[str, object], incoming: ArrayParts) -> ArrayParts:
    """Element-wise arithmetic in the input type: same shape, same type."""
    return incoming
