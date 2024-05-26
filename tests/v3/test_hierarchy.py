from __future__ import annotations

import pytest

from zarr.array import Array
from zarr.chunk_grids import RegularChunkGrid
from zarr.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.group import GroupMetadata
from zarr.hierarchy import ArrayModel, GroupModel
from zarr.metadata import ArrayV3Metadata
from zarr.store.core import StorePath
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


@pytest.mark.parametrize("attributes", ({}, {"foo": 100}))
@pytest.mark.parametrize(
    "members",
    (
        None,
        {},
        {
            "foo": ArrayModel(
                shape=(100,),
                data_type="uint8",
                chunk_grid=RegularChunkGrid(chunk_shape=(10,)),
                chunk_key_encoding=DefaultChunkKeyEncoding(),
                fill_value=0,
                attributes={"foo": 10},
            ),
            "bar": GroupModel(
                attributes={"name": "bar"},
                members={
                    "subarray": ArrayModel(
                        shape=(100,),
                        data_type="uint8",
                        chunk_grid=RegularChunkGrid(chunk_shape=(10,)),
                        chunk_key_encoding=DefaultChunkKeyEncoding(),
                        fill_value=0,
                        attributes={"foo": 10},
                    )
                },
            ),
        },
    ),
)
def test_groupmodel_to_stored(
    memory_store: MemoryStore,
    attributes: dict[str, int],
    members: None | dict[str, ArrayModel | GroupModel],
):
    model = GroupModel(attributes=attributes, members=members)
    group = model.to_stored(StorePath(memory_store, path="test"))
    model_rt = GroupModel.from_stored(group)
    assert model_rt.attributes == model.attributes
    if members is not None:
        assert model_rt.members == model.members
    else:
        assert model_rt.members == {}
