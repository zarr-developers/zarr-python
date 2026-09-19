from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from zarr.core.common import JSON

RESAVE_METADATA_HINT: Final = (
    "Re-save the array metadata to store the corrected value: open the array "
    "writable and call `array.update_attributes({})`."
)
"""How to persist metadata that was read under a compatibility policy.

`update_attributes` rewrites the whole metadata document from the parsed
(corrected) metadata, so an empty update is enough.
"""


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return dict(data)
