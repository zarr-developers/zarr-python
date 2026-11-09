"""The v2 `object` family, `|O`: Python objects. The fill value may be any JSON value."""

from __future__ import annotations

from typing import Final

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.scalar import (
    ZarrV2ByteOrder,
)


class ZarrV2ObjectConfiguration(TypedDict, closed=True):
    """The configuration that `|O` carries: a byte order, which the type ignores."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    """The byte order that the typestr writes. The canonical form writes `|`, as NumPy does."""


def _canonical(configuration: ZarrV2ObjectConfiguration) -> ZarrV2ObjectConfiguration:
    """Set the byte order to `|` (no byte order), as NumPy writes it."""
    return {**configuration, "byteorder": "|"}


OBJECT_DATA_TYPE_V2: Final = ZarrV2DataTypeDefinition(
    name="object", configuration=ZarrV2ObjectConfiguration, canonical=_canonical
)
"""`|O`: Python objects, each encoded by a filter. The fill value may be any JSON value."""

__all__ = ["OBJECT_DATA_TYPE_V2", "ZarrV2ObjectConfiguration"]
