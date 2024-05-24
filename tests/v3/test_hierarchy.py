from __future__ import annotations

from zarr.array import Array
from zarr.chunk_grids import RegularChunkGrid
from zarr.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.group import GroupMetadata
from zarr.hierarchy import ArrayModel, GroupModel
from zarr.metadata import ArrayV3Metadata
from zarr.store.memory import MemoryStore


def test_array_model_from_dict() -> None:
    array_meta = ArrayV3Metadata(
        shape=(10,),
        data_type="uint8",
        chunk_grid=RegularChunkGrid(chunk_shape=(10,)),
        chunk_key_encoding=DefaultChunkKeyEncoding(),
        fill_value=0,
        attributes={"foo": 10},
    )

    model = ArrayModel.from_dict(array_meta.to_dict())
    assert model.to_dict() == array_meta.to_dict()


def test_array_model_to_stored(memory_store: MemoryStore) -> None:
    model = ArrayModel(
        shape=(10,),
        data_type="uint8",
        chunk_grid=RegularChunkGrid(chunk_shape=(10,)),
        chunk_key_encoding=DefaultChunkKeyEncoding(),
        fill_value=0,
        attributes={"foo": 10},
    )

    array = model.to_stored(memory_store)
    assert array.metadata.to_dict() == model.to_dict()


def test_array_model_from_stored(memory_store: MemoryStore) -> None:
    array_meta = ArrayV3Metadata(
        shape=(10,),
        data_type="uint8",
        chunk_grid=RegularChunkGrid(chunk_shape=(10,)),
        chunk_key_encoding=DefaultChunkKeyEncoding(),
        fill_value=0,
        attributes={"foo": 10},
    )

    array = Array.from_dict(memory_store, array_meta.to_dict())
    array_model = ArrayModel.from_stored(array)
    assert array_model.to_dict() == array_meta.to_dict()


def test_groupmodel_from_dict() -> None:
    group_meta = GroupMetadata(attributes={"foo": "bar"})
    model = GroupModel.from_dict({**group_meta.to_dict(), "members": None})
    assert model.to_dict() == {**group_meta.to_dict(), "members": None}


def test_groupmodel_to_stored(): ...
