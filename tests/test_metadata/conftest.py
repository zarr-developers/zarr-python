from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.codecs.bytes import BytesCodec

if TYPE_CHECKING:
    from zarr_metadata.v3.chunk_grid.regular import RegularChunkGridMetadata
    from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncodingMetadata

    from zarr.core.metadata import ArrayMetadataJSON_V3


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
    # Bind chunk-grid and chunk-key-encoding subdicts to their precise
    # zarr-metadata types so structural shape errors surface here rather
    # than downstream.
    chunk_grid: RegularChunkGridMetadata = {
        "name": "regular",
        "configuration": {"chunk_shape": (4, 4)},
    }
    chunk_key_encoding: DefaultChunkKeyEncodingMetadata = {
        "name": "default",
        "configuration": {"separator": "/"},
    }
    d: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": "uint8",
        # mypy does not recognize structural subtyping between TypedDicts,
        # so `RegularChunkGridMetadata` is not seen as assignable to the
        # outer `str | NamedConfig` field type even though it is. The
        # bound variables above are correct; suppress the spurious
        # `typeddict-item` rejections here.
        "chunk_grid": chunk_grid,  # type: ignore[typeddict-item]
        "chunk_key_encoding": chunk_key_encoding,  # type: ignore[typeddict-item]
        "fill_value": 0,
        "codecs": (BytesCodec().to_dict(),),  # type: ignore[typeddict-item]
        "attributes": {},
        "storage_transformers": (),
    }
    d.update(overrides)  # type: ignore[typeddict-item]
    if extra_fields is not None:
        d.update(extra_fields)  # type: ignore[typeddict-item]
    return d
