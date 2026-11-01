"""The scope this package provides for Zarr v2."""

from __future__ import annotations

from typing import Final

from zarr_metadata.v2._context import ZarrV2Context
from zarr_metadata.v2.codec import CODECS_V2
from zarr_metadata.v2.data_type import DATA_TYPES_V2

CORE_V2_CONTEXT: Final = ZarrV2Context(data_types=DATA_TYPES_V2, codecs=CODECS_V2)
"""The data types that zarr-python 2.x writes and the codecs that numcodecs 0.16 configures."""

RENAMED_SCOPES_V2: Final = {"CORE_V2": ("CORE_V2_CONTEXT", CORE_V2_CONTEXT)}
"""The v2 scope's name before it ended in `_CONTEXT`, with the scope's name now and the scope."""

__all__ = ["CORE_V2_CONTEXT", "RENAMED_SCOPES_V2"]
