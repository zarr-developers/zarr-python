from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from zarr.core.common import JSON


def parse_attributes(data: object) -> dict[str, JSON]:
    if data is None:
        return {}
    if not isinstance(data, Mapping) or not all(isinstance(k, str) for k in data):
        msg = (
            "Attributes metadata that is not a mapping with string keys is "
            "deprecated and will be rejected in a future release. "
            f"Got {data!r}."
        )
        warnings.warn(msg, ZarrDeprecationWarning, stacklevel=2)
    return dict(cast("Mapping[str, JSON]", data))
