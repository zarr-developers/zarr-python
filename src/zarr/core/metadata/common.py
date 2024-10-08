from __future__ import annotations

from typing import TYPE_CHECKING

from .v2 import ArrayV2Metadata, ArrayV2MetadataDict
from .v3 import ArrayV3Metadata, ArrayV3MetadataDict

if TYPE_CHECKING:
    from typing import TypeAlias

    from zarr.core.common import JSON

ArrayMetadata: TypeAlias = ArrayV2Metadata | ArrayV3Metadata
ArrayMetadataDict: TypeAlias = ArrayV2MetadataDict | ArrayV3MetadataDict


def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    if data is None:
        return {}

    return data
