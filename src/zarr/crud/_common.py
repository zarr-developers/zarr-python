from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.common import JSON


def parse_array_metadata(
    metadata: Mapping[str, JSON],
) -> ArrayV3Metadata | ArrayV2Metadata:
    """Parse a metadata document into a v2 or v3 array metadata object."""
    data = dict(metadata)
    if data.get("zarr_format") == 3:
        return ArrayV3Metadata.from_dict(data)
    return ArrayV2Metadata.from_dict(data)
