import enum
import json
import warnings
from typing import Any, cast

import numcodecs
import numpy as np
import pytest
from packaging.version import Version

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.codecs import BloscCodec
from zarr.codecs.blosc import (
    BLOSC_CNAME,
    BLOSC_SHUFFLE,
    BloscCname,
    BloscCnameLiteral,
    BloscShuffle,
    BloscShuffleLiteral,
)
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt16, get_data_type_from_native_dtype
from zarr.storage import MemoryStore, StorePath


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
async def test_blosc_evolve(dtype: str) -> None:
    typesize = np.dtype(dtype).itemsize
    path = "blosc_evolve"
    store = MemoryStore()
    spath = StorePath(store, path)
    zarr.create_array(
        spath,
        shape=(16, 16),
        chunks=(16, 16),
        dtype=dtype,
        fill_value=0,
        compressors=BloscCodec(),
    )
    buf = await store.get(f"{path}/zarr.json", prototype=default_buffer_prototype())
    assert buf is not None
    zarr_json = json.loads(buf.to_bytes())
    blosc_configuration_json = zarr_json["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == typesize
    if typesize == 1:
        assert blosc_configuration_json["shuffle"] == "bitshuffle"
    else:
        assert blosc_configuration_json["shuffle"] == "shuffle"

    path2 = "blosc_evolve_sharding"
    spath2 = StorePath(store, path2)
    zarr.create_array(
        spath2,
        shape=(16, 16),
        chunks=(16, 16),
        shards=(16, 16),
        dtype=dtype,
        fill_value=0,
        compressors=BloscCodec(),
    )
    buf = await store.get(f"{path2}/zarr.json", prototype=default_buffer_prototype())
    assert buf is not None
    zarr_json = json.loads(buf.to_bytes())
    blosc_configuration_json = zarr_json["codecs"][0]["configuration"]["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == typesize
    if typesize == 1:
        assert blosc_configuration_json["shuffle"] == "bitshuffle"
    else:
        assert blosc_configuration_json["shuffle"] == "shuffle"


@pytest.mark.parametrize("shuffle", [None, "bitshuffle", "legacy-enum"])
@pytest.mark.parametrize("typesize", [None, 1, 2])
def test_tunable_attrs_param(
    shuffle: None | BloscShuffleLiteral | str, typesize: None | int
) -> None:
    """
    Test that the tunable_attrs parameter is set as expected when creating a BloscCodec.
    """
    # Materialize BloscShuffle.shuffle via the deprecation shim without
    # contaminating the BloscCodec construction below with that warning.
    if shuffle == "legacy-enum":
        with pytest.warns(DeprecationWarning, match="BloscShuffle.shuffle"):
            shuffle_arg: None | BloscShuffleLiteral | str = BloscShuffle.shuffle
    else:
        shuffle_arg = shuffle

    codec = BloscCodec(typesize=typesize, shuffle=cast(BloscShuffleLiteral | None, shuffle_arg))

    if shuffle_arg is None:
        assert codec.shuffle == "bitshuffle"  # default shuffle
        assert "shuffle" in codec._tunable_attrs
    if typesize is None:
        assert codec.typesize == 1  # default typesize
        assert "typesize" in codec._tunable_attrs

    new_dtype = UInt16()
    array_spec = ArraySpec(
        shape=(1,),
        dtype=new_dtype,
        fill_value=1,
        prototype=default_buffer_prototype(),
        config=cast(ArrayConfig, {}),
    )

    evolved_codec = codec.evolve_from_array_spec(array_spec=array_spec)
    if typesize is None:
        assert evolved_codec.typesize == new_dtype.item_size
    else:
        assert evolved_codec.typesize == codec.typesize
    if shuffle_arg is None:
        assert evolved_codec.shuffle == "shuffle"
    else:
        assert evolved_codec.shuffle == codec.shuffle


async def test_typesize() -> None:
    a = np.arange(1000000, dtype=np.uint64)
    codecs = [zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()]
    z = zarr.array(a, chunks=(10000), codecs=codecs)
    data = await z.store.get("c/0", prototype=default_buffer_prototype())
    assert data is not None
    bytes = data.to_bytes()
    size = len(bytes)
    msg = f"Blosc size mismatch.  First 10 bytes: {bytes[:20]!r} and last 10 bytes: {bytes[-20:]!r}"
    if Version(numcodecs.__version__) >= Version("0.16.0"):
        expected_size = 402
        assert size == expected_size, msg
    else:
        expected_size = 10216
    assert size == expected_size, msg


def test_blosc_codec_supports_sync() -> None:
    assert isinstance(BloscCodec(), SupportsSyncCodec)


def test_blosc_codec_sync_roundtrip() -> None:
    codec = BloscCodec(typesize=8)
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
    np.testing.assert_array_equal(arr, result)


@pytest.mark.parametrize("cname", BLOSC_CNAME)
def test_blosc_codec_accepts_all_cnames(cname: BloscCnameLiteral) -> None:
    """
    Every compressor name in BLOSC_CNAME is accepted by BloscCodec and round-trips
    to the same value on the stored attribute. Adding a new value to the
    BloscCnameLiteral type alias without also adding it to BLOSC_CNAME (or vice
    versa) is caught here.
    """
    codec = BloscCodec(cname=cname)
    assert codec.cname == cname


@pytest.mark.parametrize("shuffle", BLOSC_SHUFFLE)
def test_blosc_codec_accepts_all_shuffles(shuffle: BloscShuffleLiteral) -> None:
    """
    Every shuffle mode in BLOSC_SHUFFLE is accepted by BloscCodec and round-trips
    to the same value on the stored attribute. Adding a new value to the
    BloscShuffleLiteral type alias without also adding it to BLOSC_SHUFFLE (or
    vice versa) is caught here.
    """
    codec = BloscCodec(shuffle=shuffle)
    assert codec.shuffle == shuffle


@pytest.mark.parametrize("shuffle", BLOSC_SHUFFLE)
@pytest.mark.parametrize("cname", BLOSC_CNAME)
def test_blosc_codec_json_roundtrip(cname: BloscCnameLiteral, shuffle: BloscShuffleLiteral) -> None:
    """
    JSON serialization (to_dict / from_dict) preserves every (cname, shuffle)
    pair drawn from BLOSC_CNAME x BLOSC_SHUFFLE. Guards against drift in the
    codec's V3 JSON form for any combination of compressor and shuffle option.

    The non-varied fields are fully specified so the codec has no tunable
    attributes; tunability is not part of the JSON form and would otherwise
    cause spurious round-trip mismatches.
    """
    codec = BloscCodec(typesize=1, cname=cname, clevel=5, shuffle=shuffle, blocksize=0)
    restored = BloscCodec.from_dict(codec.to_dict())
    assert restored == codec


@pytest.mark.parametrize(
    ("enum_cls", "member", "expected"),
    [
        (BloscShuffle, "shuffle", "shuffle"),
        (BloscCname, "zstd", "zstd"),
    ],
)
def test_blosc_enum_member_access_warns(enum_cls: type, member: str, expected: str) -> None:
    """
    Accessing a member on the deprecated BloscShuffle / BloscCname classes
    emits a DeprecationWarning and resolves to the equivalent literal string.
    """
    match = f"{enum_cls.__name__}.{member}"
    with pytest.warns(DeprecationWarning, match=match):
        value = getattr(enum_cls, member)
    assert value == expected


def test_blosc_enum_classes_import_silently() -> None:
    """
    Importing the deprecated enum classes by name must not emit a warning;
    only member access does. This guards against the blosc module accidentally
    triggering its own deprecation warnings when it (or zarr) is imported.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from zarr.codecs.blosc import (  # pylint: disable=reimported
            BloscCname as _BloscCname,  # noqa: F401
        )
        from zarr.codecs.blosc import (  # pylint: disable=reimported
            BloscShuffle as _BloscShuffle,  # noqa: F401
        )


def test_blosc_codec_init_with_enum_instance_warns() -> None:
    """
    Passing a real `enum.Enum` instance to BloscCodec.__init__ (e.g. an
    instance materialized before the deprecation shim was introduced) must
    trigger the init-level deprecation warning and still normalize the value
    to the corresponding literal string.
    """

    class LegacyShuffle(enum.Enum):
        bitshuffle = "bitshuffle"

    class LegacyCname(enum.Enum):
        zstd = "zstd"

    with pytest.warns(DeprecationWarning, match="enum"):
        codec = BloscCodec(
            cname=cast(BloscCname, LegacyCname.zstd),
            shuffle=cast(BloscShuffle, LegacyShuffle.bitshuffle),
        )
    assert codec.cname == "zstd"
    assert codec.shuffle == "bitshuffle"


@pytest.mark.parametrize("param", ["cname", "shuffle"])
def test_blosc_codec_rejects_unknown(param: str) -> None:
    """
    BloscCodec.__init__ raises ValueError when given a string outside the
    allowed set for `cname` or `shuffle`, and the error message names the
    offending parameter.
    """
    kwargs: dict[str, Any] = {param: f"not-a-{param}"}
    with pytest.raises(ValueError, match=f"{param} must be one of"):
        BloscCodec(**kwargs)


@pytest.mark.parametrize("enum_cls", [BloscShuffle, BloscCname])
def test_blosc_enum_attribute_error_for_unknown_member(enum_cls: type) -> None:
    """
    Attribute access for a name that is not a known member of the deprecated
    enum classes falls through to AttributeError, matching the behavior of a
    regular class.
    """
    unknown_name = "not_a_member"
    with pytest.raises(AttributeError):
        getattr(enum_cls, unknown_name)
