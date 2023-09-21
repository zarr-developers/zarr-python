import pytest
import json

from zarr.zoms import (
    BaseZOM,
    BaseV2ZOM,
    BaseV3ZOM,
    V2ArrayZOM,
    V2GroupZOM,
    V3ArrayZOM,
    V3ChunkGrid,
    V3ChunkKeyEncoding,
    V3ChunkKeyEncodingConfiguration,
    V3GroupZOM,
    V3RegularChunkConfiguration,
)
import zarr


@pytest.fixture()
def v2array() -> V2ArrayZOM:

    return V2ArrayZOM(
        shape=(10, 10),
        chunks=(2, 2),
        dtype="<f8",
        compressor=None,
        fill_value=-9999.0,
        order="C",
        filters=None,
        attributes={"foo": "bar"},
    )


@pytest.fixture()
def v2group() -> V2GroupZOM:

    return V2GroupZOM(
        attributes={"foo": "bar"},
    )


@pytest.fixture()
def v3array() -> V3ArrayZOM:

    return V3ArrayZOM(
        shape=(10, 10),
        chunk_grid=V3ChunkGrid(
            name="default", configuration=V3RegularChunkConfiguration(chunk_shape=(2, 2))
        ),
        chunk_key_encoding=V3ChunkKeyEncoding(
            "default", configuration=V3ChunkKeyEncodingConfiguration(separator="/")
        ),
        dtype="<f8",
        fill_value=-9999.0,
        codecs=[],  # [{"name": "gzip", "configuration": {"level": 1}}]
        storage_transformers=[],
        attributes={"foo": "bar"},
    )


@pytest.fixture()
def v3group() -> V3GroupZOM:

    return V3GroupZOM(
        attributes={"foo": "bar"},
    )


def test_v2_array_is_zom(v2array) -> None:
    assert isinstance(v2array, BaseZOM)
    assert isinstance(v2array, BaseV2ZOM)
    assert isinstance(v2array, V2ArrayZOM)


def test_v2_array_contents(v2array) -> None:

    jsons = v2array.serialize()

    # .zarray
    array_json = jsons[".zarray"]
    assert isinstance(array_json, bytes)
    loaded = json.loads(array_json)
    expected_keys = [
        "shape",
        "shape",
        "dtype",
        "compressor",
        "fill_value",
        "order",
        "filters",
        "zarr_format",
    ]
    for key in expected_keys:
        assert key in loaded
    unexpected_keys = ["attrs", "members"]
    for key in unexpected_keys:
        assert key not in loaded

    # .zattrs
    attrs_json = jsons[".zattrs"]
    assert isinstance(attrs_json, bytes)
    loaded = json.loads(attrs_json)
    assert loaded == v2array.attributes


def test_v2_array_roundtrip(v2array) -> None:
    jsons = v2array.serialize()
    new = V2ArrayZOM.deserialize(jsons)
    assert new == v2array


def test_v2_array_interop(v2array) -> None:

    store = zarr.MemoryStore()
    jsons = v2array.serialize()
    store.update(jsons)

    # open using open_array
    arr = zarr.open_array(store=store, mode="r", zarr_version=v2array.zarr_format)
    assert dict(arr.attrs) == v2array.attributes
    assert arr.shape == v2array.shape
    assert arr.chunks == v2array.chunks
    assert arr.dtype == v2array.dtype
    assert arr.fill_value == v2array.fill_value
    assert arr.order == v2array.order
    assert arr.filters == v2array.filters

    # deserialize after creating array with open_array
    store = zarr.MemoryStore()
    arr = zarr.open_array(
        store=store,
        mode="w",
        shape=v2array.shape,
        chunks=v2array.chunks,
        dtype=v2array.dtype,
        compressor=v2array.compressor,
        fill_value=v2array.fill_value,
        filters=v2array.filters,
        order=v2array.order,
        # Note: unfortunate that zarr_version and zarr_format
        zarr_version=v2array.zarr_format,
    )
    arr.attrs.put(v2array.attributes)
    model = V2ArrayZOM.deserialize(dict(store))
    assert model == v2array


def test_v2_group_is_zom(v2group) -> None:
    assert isinstance(v2group, BaseZOM)
    assert isinstance(v2group, BaseV2ZOM)
    assert isinstance(v2group, V2GroupZOM)


def test_v2_group_contents(v2group) -> None:

    jsons = v2group.serialize()

    # .zgroup
    group_json = jsons[".zgroup"]
    assert isinstance(group_json, bytes)
    loaded = json.loads(group_json)
    assert loaded == {"zarr_format": 2}

    # .zattrs
    attrs_json = jsons[".zattrs"]
    assert isinstance(attrs_json, bytes)
    loaded = json.loads(attrs_json)
    assert loaded == v2group.attributes


def test_v2_group_roundtrip(v2group) -> None:
    jsons = v2group.serialize()
    new = V2GroupZOM.deserialize(jsons)
    assert new == v2group


def test_v2_group_interop(v2group) -> None:

    store = zarr.MemoryStore()
    jsons = v2group.serialize()
    store.update(jsons)

    # open using open_group
    group = zarr.open_group(store=store, mode="r", zarr_version=v2group.zarr_format)
    assert dict(group.attrs) == v2group.attributes

    store = zarr.MemoryStore()
    group = zarr.open_group(store=store, mode="w", zarr_version=v2group.zarr_format)
    group.attrs.put(v2group.attributes)
    model = V2GroupZOM.deserialize(dict(store))
    assert model == v2group


def test_v3_array_is_zom(v3array) -> None:
    assert isinstance(v3array, BaseZOM)
    assert isinstance(v3array, BaseV3ZOM)
    assert isinstance(v3array, V3ArrayZOM)


def test_v3_array_contents(v3array) -> None:

    jsons = v3array.serialize()

    # zarr.json
    array_json = jsons["zarr.json"]
    assert isinstance(array_json, bytes)
    loaded = json.loads(array_json)
    expected_keys = [
        "shape",
        "dtype",
        "fill_value",
        "zarr_format",
        "attributes",
    ]
    for key in expected_keys:
        assert key in loaded
    unexpected_keys = ["members"]
    for key in unexpected_keys:
        assert key not in loaded
    assert loaded["attributes"] == v3array.attributes


@pytest.mark.xfail(reason="need to unpack nested metadata in deserialize")
def test_v3_array_roundtrip(v3array) -> None:
    jsons = v3array.serialize()
    new = V3ArrayZOM.deserialize(jsons)
    assert new == v3array


@pytest.mark.xfail(reason="need to update v3 group schema for final spec")
def test_v3_array_interop(v3array) -> None:

    store = zarr.MemoryStoreV3()
    jsons = v3array.serialize()
    store.update(jsons)

    # open using open_array
    arr = zarr.open_array(store=store, mode="r", zarr_version=v3array.zarr_format)
    assert dict(arr.attrs) == v3array.attrs
    assert arr.shape == v3array.shape
    assert arr.dtype == v3array.dtype
    assert arr.fill_value == v3array.fill_value
    # TODO: add other v3 parameters

    # deserialize after creating array with open_array
    store = zarr.MemoryStoreV3()
    arr = zarr.open_array(
        store=store,
        mode="w",
        shape=v3array.shape,
        # TODO: explore how the high-level zarr api can adapt to the v3-specific config
        # chunks=v3array.chunks,
        dtype=v3array.dtype,
        # compressor=v3array.compressor,
        fill_value=v3array.fill_value,
        # filters=v3array.filters,
        order=v3array.order,
        # Note: unfortunate that zarr_version and zarr_format
        zarr_version=v3array.zarr_format,
    )
    arr.attrs.put(v3array.attributes)
    model = V3ArrayZOM.deserialize(dict(store))
    assert model == v3array


def test_v3_group_is_zom(v3group) -> None:
    assert isinstance(v3group, BaseZOM)
    assert isinstance(v3group, BaseV3ZOM)
    assert isinstance(v3group, V3GroupZOM)


def test_v3_group_contents(v3group) -> None:

    jsons = v3group.serialize()

    # zarr.json
    group_json = jsons["zarr.json"]
    assert isinstance(group_json, bytes)
    loaded = json.loads(group_json)
    assert loaded == {"zarr_format": 3, "node_type": "group", "attributes": v3group.attributes}


def test_v3_group_roundtrip(v3group) -> None:
    jsons = v3group.serialize()
    new = V3GroupZOM.deserialize(jsons)
    assert new == v3group


@pytest.mark.xfail(reason="need to update v3 group schema for final spec")
def test_v3_group_interop(v3group) -> None:

    store = zarr.MemoryStoreV3()
    jsons = v3group.serialize()
    store.update(jsons)
    print(store["zarr.json"])

    # open using open_group
    group = zarr.open_group(store=store, mode="r", zarr_version=v3group.zarr_format)
    assert dict(group.attrs) == v3group.attributes

    store = zarr.MemoryStoreV3()
    group = zarr.open_group(store=store, mode="w", zarr_version=v3group.zarr_format)
    group.attrs.put(v3group.attributes)
    model = V3GroupZOM.deserialize(dict(store))
    assert model == v3group
