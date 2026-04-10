from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.codecs.bytes import BytesCodec

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import ArrayMetadataJSON_V3


def minimal_metadata_dict_v3(
    extra_fields: dict[str, Any] | None = None, **overrides: Any
) -> ArrayMetadataJSON_V3:
    """Build a minimal valid V3 array metadata JSON dict.

    The output matches the shape of ``ArrayV3Metadata.to_dict()`` — all
    fields that ``to_dict`` always emits are included.

    Parameters
    ----------
    extra_fields : dict, optional
        Extra keys to inject into the dict (e.g. extension fields).
    **overrides
        Override any of the standard metadata fields.
    """
    d: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": (BytesCodec().to_dict(),),
        "attributes": {},
        "storage_transformers": (),
    }
    d.update(overrides)
    if extra_fields is not None:
        d.update(extra_fields)
    return d
