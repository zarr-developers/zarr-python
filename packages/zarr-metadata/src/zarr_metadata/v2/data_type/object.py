"""The v2 `object` family, `|O`: Python objects, whose fill value is any JSON."""

from __future__ import annotations

from typing import Final, cast

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.scalar import (
    ZarrV2ByteOrder,  # noqa: TC001 - a TypedDict's annotations are evaluated at run time
)


class ZarrV2ObjectConfiguration(TypedDict, closed=True):
    """What `|O` carries: a byte order, which the type ignores."""

    byteorder: ReadOnly[ZarrV2ByteOrder]


def _canonical(configuration: ZarrV2ObjectConfiguration) -> ZarrV2ObjectConfiguration:
    """No byte order, `|`, as NumPy writes it."""
    return cast("ZarrV2ObjectConfiguration", {**configuration, "byteorder": "|"})


OBJECT_V2: Final = ZarrV2DataTypeDefinition(
    name="object", configuration=ZarrV2ObjectConfiguration, canonical=_canonical
)
"""`|O`: Python objects, each encoded by a filter; the fill value any JSON."""

__all__ = ["OBJECT_V2", "ZarrV2ObjectConfiguration"]
