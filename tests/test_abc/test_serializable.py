from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.dtype.npy.int import UInt8
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from zarr.abc.serializable import JSONSerializable
    from zarr.core.metadata.v3 import ArrayMetadataJSON_V3


def test_array_v3_metadata_to_json() -> None:
    """
    ArrayV3Metadata satisfies the JSONSerializable protocol parameterized
    on its JSON output type, and ``to_json`` returns the same payload as
    ``to_dict``.
    """
    metadata = ArrayV3Metadata(
        shape=(10,),
        data_type=UInt8(),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0,
        codecs=({"name": "bytes", "configuration": {"endian": "little"}},),
        attributes={},
        dimension_names=None,
    )
    serializable: JSONSerializable[ArrayMetadataJSON_V3] = metadata
    assert serializable.to_json() == metadata.to_dict()
