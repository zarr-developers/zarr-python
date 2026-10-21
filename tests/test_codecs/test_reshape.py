import json

import numpy as np
import pytest

from zarr.codecs.reshape import ReshapeCodec, ShapeSpec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt8
from zarr.core.metadata.v3 import RegularChunkGridMetadata


def test_deserialize() -> None:
    s = '{"name": "reshape", "configuration": {"shape": [5, [1, 2], -1]}}'
    d = json.loads(s)
    ReshapeCodec.from_dict(d)


@pytest.mark.asyncio
async def test_encode() -> None:
    arr_flat = np.arange(60, dtype="uint8")
    arr_3x2 = np.reshape(arr_flat, (3, 4, 5))
    dt = UInt8()
    codec = ReshapeCodec(shape=((0, 1), 5))
    proto = default_buffer_prototype()
    (encoded, *_) = await codec.encode(
        [
            (
                proto.nd_buffer.from_numpy_array(arr_3x2),
                ArraySpec(
                    arr_3x2.shape,
                    dt,
                    fill_value=0,
                    config=ArrayConfig("C", False),
                    prototype=proto,
                ),
            )
        ]
    )
    assert encoded is not None
    assert encoded.shape == (12, 5)
    assert np.all(np.equal(np.ravel(encoded.as_numpy_array()), arr_flat))


@pytest.mark.asyncio
async def test_decode() -> None:
    arr_flat = np.arange(60, dtype="uint8")
    arr_12x5 = np.reshape(arr_flat, (12, 5))
    dt = UInt8()
    codec = ReshapeCodec(shape=((0, 1), 5))
    proto = default_buffer_prototype()
    (encoded, *_) = await codec.decode(
        [
            (
                proto.nd_buffer.from_numpy_array(arr_12x5),
                ArraySpec(
                    (3, 4, 5),
                    dt,
                    fill_value=0,
                    config=ArrayConfig("C", False),
                    prototype=proto,
                ),
            )
        ]
    )
    assert encoded is not None
    assert encoded.shape == (3, 4, 5)
    assert np.all(np.equal(np.ravel(encoded.as_numpy_array()), arr_flat))


@pytest.mark.parametrize(
    "shape_spec",
    [
        (-2,),  # negative number isn't -1
        (-1, -1),  # more than one -1
        ((-2, 1),),  # negative index
        ((1, 0),),  # decreasing
        # (((0, 4),), 3),  # missing dim
    ],
)
def test_invalid_shape_spec(shape_spec: ShapeSpec) -> None:
    with pytest.raises(ValueError):
        ReshapeCodec(shape=shape_spec)


def test_compute_encoded_size() -> None:
    c = ReshapeCodec(shape=((0, 1), 5))

    dec_shape = (64, 64)
    dec_size = np.prod(dec_shape).item()
    enc_size = c.compute_encoded_size(
        dec_size,
        # ignored
        ArraySpec(
            (64, 64),
            UInt8(),
            fill_value=0,
            config=ArrayConfig("C", False),
            prototype=default_buffer_prototype(),
        ),
    )
    assert dec_size == enc_size


@pytest.mark.parametrize(
    ("shape", "shape_spec"),
    [
        ((3, 4), ((0, 1, 2),)),  # index out of range
        ((3, 4), (5, -1)),  # indivisible -1
        ((3, 4), (5, 2)),  # wrong shape
    ],
)
def test_validate_invalid(shape: tuple[int, ...], shape_spec: ShapeSpec) -> None:
    """Stand-in for _compute_output_shape"""
    codec = ReshapeCodec(shape=shape_spec)
    with pytest.raises(ValueError):
        codec.validate(
            shape,
            UInt8(),  # ignored
            RegularChunkGridMetadata(chunk_shape=shape),
        )


@pytest.mark.parametrize(
    ("input_shape", "shape_spec", "output_shape"),
    [
        ((3, 4), (4, 3), (4, 3)),
        ((3, 4), (6, -1), (6, 2)),
        ((3, 4), ((0, 1),), (12,)),
    ],
)
def test_valid(
    input_shape: tuple[int, ...], shape_spec: ShapeSpec, output_shape: tuple[int, ...]
) -> None:
    codec = ReshapeCodec(shape=shape_spec)
    dtype = UInt8()
    codec.validate(input_shape, dtype, RegularChunkGridMetadata(chunk_shape=input_shape))
    input_spec = ArraySpec(
        input_shape, dtype, 0, ArrayConfig("C", False), default_buffer_prototype()
    )
    output_spec = codec.resolve_metadata(input_spec)

    assert output_spec.dtype == input_spec.dtype
    assert output_spec.config == input_spec.config
    assert output_spec.fill_value == input_spec.fill_value
    assert output_spec.prototype == input_spec.prototype

    assert output_spec.shape == output_shape


@pytest.mark.asyncio
async def test_example() -> None:
    import json

    import zarr
    from zarr.storage import ManagedMemoryStore

    # Create an array with the reshape codec
    # This reshapes chunks from (100, 50, 64, 3) to (5000, 64, 3)
    store = ManagedMemoryStore()
    _ = zarr.create_array(
        store,
        shape=(1000, 500, 64, 3),
        chunks=(100, 50, 64, 3),
        dtype="float32",
        filters=[ReshapeCodec(shape=((0, 1), (2,), 3))],
    )
    b = await store.get("zarr.json")
    assert b is not None
    s = b.to_bytes().decode("utf-8")
    d = json.loads(s)
    assert d["codecs"][0] == {
        "name": "reshape",
        "configuration": {"shape": [[0, 1], [2], 3]},
    }
